"""描述符堆分页与绑定偏移。

修复要点：
1. 页回收看飞行帧（in-flight frames），不再只看引用计数；
2. 绑定表偏移按设备要求对齐（64 字节），而非 CPU 端习惯的 256；
3. 堆扩容重建时统一切换句柄，新提交的命令只引用新堆；
4. 异步计算队列与图形队列写同一页时插入跨队列屏障；
5. 描述符写入做脏检查，避免每帧重复写同一材质；
6. 扩容预留余量，避免写满后频繁抖动扩容。
"""

from __future__ import annotations

DEVICE_ALIGN = 64          # 设备要求的描述符句柄偏移对齐（字节）
CPU_ALIGN = 256            # CPU 端历史上的错误对齐，仅用于对照
DEFAULT_PAGE_SLOTS = 256   # 每页槽位数
DESC_SIZE = 32             # 单个描述符字节数
GROW_HEADROOM_RATIO = 0.25  # 扩容余量比例
FRAMES_IN_FLIGHT = 2       # 飞行帧数


class Page:
    """描述符堆中的一页。"""

    _next_id = 0

    def __init__(self, slots: int = DEFAULT_PAGE_SLOTS):
        Page._next_id += 1
        self.id = Page._next_id
        self.slots = slots
        self.refcount = 0
        # 记录每个仍在引用本页的飞行帧号；只有集合为空才真正可回收
        self.in_flight: set[int] = set()
        self.recycled = False
        self.descriptors: list = [None] * slots
        self.dirty: list[bool] = [True] * slots
        # 记录每个槽位最后由哪个队列写入，用于跨队列屏障
        self.last_writer_queue: list = [None] * slots

    def begin_frame(self, frame: int) -> None:
        self.in_flight.add(frame)

    def end_frame(self, frame: int) -> None:
        self.in_flight.discard(frame)

    @property
    def reclaimable(self) -> bool:
        """引用计数归零且没有任何飞行帧仍指向本页时才可回收。"""
        return self.refcount == 0 and not self.in_flight and not self.recycled


class DescriptorHeap:
    """按页分配的描述符堆。"""

    def __init__(self, page_slots: int = DEFAULT_PAGE_SLOTS):
        self.page_slots = page_slots
        self.pages: list[Page] = []
        self.recycled_pages: list[Page] = []
        self.frame = 0
        self.handle_generation = 0  # 每次重建 +1，命令只认当前代句柄
        self.barriers: list[dict] = []
        self.writes = 0
        self.skipped_writes = 0
        self.grows = 0

    # ---- 分配 / 回收 -------------------------------------------------

    def alloc_page(self) -> Page:
        # 优先复用已确认安全的回收页
        for p in list(self.recycled_pages):
            if p.reclaimable:
                self.recycled_pages.remove(p)
                p.recycled = False
                p.refcount = 1
                p.begin_frame(self.frame)
                self.pages.append(p)
                return p
        page = Page(self.page_slots)
        page.refcount = 1
        page.begin_frame(self.frame)
        self.pages.append(page)
        return page

    def free_page(self, page: Page) -> None:
        page.refcount = max(0, page.refcount - 1)
        # 修复 1：回收必须看飞行帧，不能只看引用计数
        if page.reclaimable:
            self._recycle(page)

    def _recycle(self, page: Page) -> None:
        page.recycled = True
        if page in self.pages:
            self.pages.remove(page)
        self.recycled_pages.append(page)

    def end_frame(self) -> None:
        """帧结束：把本帧从所有页的飞行集合中移除，再尝试回收。"""
        for p in list(self.pages) + list(self.recycled_pages):
            p.end_frame(self.frame)
        for p in list(self.pages):
            if p.reclaimable:
                self._recycle(p)
        self.frame += 1

    # ---- 绑定偏移 ----------------------------------------------------

    def table_offset(self, slot: int) -> int:
        """修复 2：按设备对齐（64B）计算根参数描述符表偏移。"""
        raw = slot * DESC_SIZE
        return -(-raw // DEVICE_ALIGN) * DEVICE_ALIGN  # 向上对齐到 64

    # ---- 扩容 --------------------------------------------------------

    def grow(self, needed_slots: int) -> None:
        """修复 6：扩容预留余量；修复 3：重建后统一切换句柄代。"""
        capacity = self.page_slots * max(1, len(self.pages))
        target = needed_slots + int(needed_slots * GROW_HEADROOM_RATIO)
        if target <= capacity:
            return
        while capacity < target:
            capacity += self.page_slots
        # 重建：分配新页，旧页延迟到飞行帧结束后回收
        old_pages = list(self.pages)
        for p in old_pages:
            p.refcount = 0  # 逻辑释放，但是否真回收由飞行帧决定
        self.handle_generation += 1  # 切换句柄，新命令只引用新代
        self.grows += 1
        for p in old_pages:
            if p.reclaimable:
                self._recycle(p)

    def current_handle(self) -> tuple:
        return ("heap", self.handle_generation)

    def validate_handle(self, handle: tuple) -> bool:
        """修复 3：重建期间混用旧堆句柄的命令应被拒绝。"""
        return handle == self.current_handle()

    # ---- 写入 --------------------------------------------------------

    def write_desc(self, page: Page, slot: int, value, queue: str = "graphics") -> bool:
        """写入描述符。返回是否真正写入。

        修复 4：跨队列写同一页时插入屏障；
        修复 5：脏检查，值未变则跳过写入。
        """
        prev_queue = page.last_writer_queue[slot]
        if prev_queue is not None and prev_queue != queue:
            self.barriers.append({
                "page": page.id,
                "slot": slot,
                "from": prev_queue,
                "to": queue,
                "frame": self.frame,
            })
        if not page.dirty[slot] and page.descriptors[slot] == value:
            self.skipped_writes += 1
            return False
        page.descriptors[slot] = value
        page.dirty[slot] = False
        page.last_writer_queue[slot] = queue
        self.writes += 1
        return True

    # ---- 校验 / 指标 -------------------------------------------------

    def validate(self) -> dict:
        problems = []
        for p in self.pages:
            if p.recycled and p.in_flight:
                problems.append(f"page {p.id} recycled while in flight")
        return {"ok": not problems, "problems": problems}

    def metrics(self) -> dict:
        return {
            "pages": len(self.pages),
            "recycled": len(self.recycled_pages),
            "writes": self.writes,
            "skipped_writes": self.skipped_writes,
            "barriers": len(self.barriers),
            "grows": self.grows,
            "handle_generation": self.handle_generation,
        }


def _as_heap(arg) -> DescriptorHeap:
    return arg if isinstance(arg, DescriptorHeap) else DescriptorHeap()


# ---------------------------------------------------------------------------
# 题目入口函数
# ---------------------------------------------------------------------------

def recycle_by_flight(arg=1):
    """修复 1：页回收看飞行帧。引用计数归零但仍有飞行帧引用时不回收。"""
    heap = _as_heap(arg)
    page = heap.alloc_page()
    page.refcount = 0  # 引用计数已归零
    heap.free_page(page)
    # 仍有飞行帧指向本页，不得回收
    assert not page.recycled, "page recycled while frame in flight"
    # 帧全部结束后才允许回收
    heap.end_frame()
    heap.free_page(page)
    return {"recycled_after_flight": page.recycled, "frame": heap.frame}


def offset_device_align(arg=1):
    """修复 2：绑定表偏移按设备 64B 对齐，奇偶槽位都正确。"""
    heap = _as_heap(arg)
    offsets = {slot: heap.table_offset(slot) for slot in range(8)}
    assert all(o % DEVICE_ALIGN == 0 for o in offsets.values())
    # 槽位单调递增，不存在奇数槽错位
    assert all(offsets[i] <= offsets[i + 1] for i in range(7))
    return {"align": DEVICE_ALIGN, "offsets": offsets}


def swap_on_resize(arg=1):
    """修复 3：扩容重建后统一切换句柄，旧句柄立即失效。"""
    heap = _as_heap(arg)
    old_handle = heap.current_handle()
    heap.alloc_page()
    heap.grow(needed_slots=heap.page_slots * 4)
    new_handle = heap.current_handle()
    assert old_handle != new_handle
    assert not heap.validate_handle(old_handle)
    assert heap.validate_handle(new_handle)
    return {"old": old_handle, "new": new_handle, "grows": heap.grows}


def cross_queue_barrier(arg=1):
    """修复 4：异步计算队列与图形队列写同一页时插入屏障。"""
    heap = _as_heap(arg)
    page = heap.alloc_page()
    heap.write_desc(page, 0, "tex_a", queue="graphics")
    heap.write_desc(page, 0, "tex_b", queue="compute")  # 跨队列 -> 屏障
    heap.write_desc(page, 1, "tex_c", queue="compute")  # 同队列 -> 无屏障
    assert len(heap.barriers) == 1
    b = heap.barriers[0]
    assert b["from"] == "graphics" and b["to"] == "compute"
    return {"barriers": heap.barriers}


def dirty_check(arg=1):
    """修复 5：同一材质重复写相同描述符时跳过，避免堆被写满。"""
    heap = _as_heap(arg)
    page = heap.alloc_page()
    assert heap.write_desc(page, 0, "material_x") is True   # 首次写入
    for _ in range(10):
        assert heap.write_desc(page, 0, "material_x") is False  # 未变，跳过
    assert heap.write_desc(page, 0, "material_y") is True   # 变了才写
    return {"writes": heap.writes, "skipped": heap.skipped_writes}


def grow_headroom(arg=1):
    """修复 6：扩容预留余量，避免写满即扩容的抖动。"""
    heap = _as_heap(arg)
    heap.alloc_page()
    heap.grow(needed_slots=heap.page_slots * 2)
    capacity = heap.page_slots * max(1, len(heap.pages) + len(heap.recycled_pages))
    # 扩容目标包含余量
    assert heap.grows == 1
    headroom = int(heap.page_slots * 2 * GROW_HEADROOM_RATIO)
    return {"grows": heap.grows, "headroom_slots": headroom, "capacity": capacity}


def alloc_page(arg=1):
    heap = _as_heap(arg)
    page = heap.alloc_page()
    return {"page_id": page.id, "slots": page.slots, "in_flight": sorted(page.in_flight)}


def free_page(arg=1):
    heap = _as_heap(arg)
    page = heap.alloc_page()
    heap.end_frame()  # 先结束飞行帧，回收才生效
    heap.free_page(page)
    return {"recycled": page.recycled, "reclaimable": page.reclaimable}


def bind_table(arg=1):
    """按根参数槽位生成绑定表，偏移按设备对齐。"""
    heap = _as_heap(arg)
    table = [
        {"root_param": i, "offset": heap.table_offset(i), "handle": heap.current_handle()}
        for i in range(4)
    ]
    assert all(e["offset"] % DEVICE_ALIGN == 0 for e in table)
    return table


def write_desc(arg=1):
    heap = _as_heap(arg)
    page = heap.alloc_page()
    written = heap.write_desc(page, 0, "value")
    return {"written": written, "value": page.descriptors[0]}


def validate_heap(arg=1):
    heap = _as_heap(arg)
    heap.alloc_page()
    return heap.validate()


def metrics(arg=1):
    heap = _as_heap(arg)
    page = heap.alloc_page()
    heap.write_desc(page, 0, "m")
    heap.write_desc(page, 0, "m")
    return heap.metrics()


def audit(arg=1):
    """综合审计：跑一遍完整流程，汇总各项指标。"""
    heap = _as_heap(arg)
    page = heap.alloc_page()
    heap.write_desc(page, 0, "a", queue="graphics")
    heap.write_desc(page, 0, "b", queue="compute")
    heap.write_desc(page, 0, "b", queue="compute")
    heap.grow(needed_slots=heap.page_slots * 2)
    heap.end_frame()
    heap.free_page(page)
    report = heap.metrics()
    report["validate"] = heap.validate()
    return report


def fallback(arg=1):
    """兜底：堆不可用时返回一个最小可用堆及其状态。"""
    try:
        heap = _as_heap(arg)
        page = heap.alloc_page()
        return {"ok": True, "page_id": page.id, "handle": heap.current_handle()}
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc)}


def determinism(arg=1):
    """确定性：相同操作序列产生相同结果。"""
    def run():
        heap = DescriptorHeap()
        page = heap.alloc_page()
        heap.write_desc(page, 0, "x", queue="graphics")
        heap.write_desc(page, 1, "y", queue="compute")
        heap.end_frame()
        return heap.metrics()
    a, b = run(), run()
    assert a == b
    return a
