from config import MIN_KEEP, MIN_SCORE, REL_RATIO


def build_site_meta(records, total_records=None):
    """从向量记录里推断站点元数据。

    ``latest_issue`` 来自 ``source_file`` 形如 ``issue-N.md`` 的最大 N；
    ``latest_date`` 是该 N 对应的 ``date`` 字段。``total_records`` 优先
    使用入参，否则用 records 长度（适合本地全量场景）。
    """
    issue_info = {}
    record_count = 0

    for record in records:
        record_count += 1
        source_file = record.get("source_file", "")
        if source_file.startswith("issue-") and source_file.endswith(".md"):
            try:
                issue_number = int(source_file[6:-3])
            except ValueError:
                continue
            issue_info[issue_number] = record.get("date", "")

    if not issue_info:
        return {
            "latest_issue": None,
            "latest_date": "",
            "total_records": total_records if total_records is not None else record_count,
        }

    latest_issue = max(issue_info)
    return {
        "latest_issue": latest_issue,
        "latest_date": issue_info[latest_issue],
        "total_records": total_records if total_records is not None else record_count,
    }


def select_results(candidates):
    """从召回候选里挑出最终展示项，按三档阈值收敛截断点：

    1. 绝对阈值 ``MIN_SCORE``：先找出所有 >= 该分数的位置，若非空则把
       截断点设在该位置之后；否则进入 fallback，整段取前 ``MIN_KEEP``。
    2. 相对阈值 ``REL_RATIO``：用 ``scores[0] * REL_RATIO`` 收紧到头部
       高分簇，跳过尾部明显掉档的项。
    3. 最终 ``cutoff`` 被夹在 ``[MIN_KEEP, len(candidates)]`` 之间。

    返回 ``(items, scores, cutoff, fallback)``，便于调用方做日志观测。
    """
    if not candidates:
        return [], [], 0, True

    scores = [score for score, _ in candidates]
    above_absolute = [index for index, score in enumerate(scores) if score >= MIN_SCORE]
    fallback = not above_absolute
    cutoff = MIN_KEEP if fallback else above_absolute[-1] + 1

    relative_threshold = scores[0] * REL_RATIO
    above_relative = [
        index for index, score in enumerate(scores) if score >= relative_threshold
    ]
    if above_relative:
        cutoff = min(cutoff, above_relative[-1] + 1)

    cutoff = max(MIN_KEEP, min(cutoff, len(candidates)))
    items = []
    for score, payload in candidates[:cutoff]:
        items.append(
            {
                "score": float(score),
                "title": payload.get("title", ""),
                "url": payload.get("url") or "",
                "source_file": payload.get("source_file", ""),
                "date": payload.get("date", ""),
                "description": (payload.get("description") or "")[:160],
            }
        )

    return items, scores, cutoff, fallback