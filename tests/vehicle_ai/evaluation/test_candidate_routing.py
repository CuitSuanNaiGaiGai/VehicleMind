from modules.vehicle_ai.context.context_selector import ContextSelector, ContextTopic


def test_candidate_queries_route_driver_and_road_domains() -> None:
    selector = ContextSelector()
    for text in (
        "我有点分心，前方有行人吗？",
        "驾驶员在吗？车现在是什么状态？",
        "现在呢？请根据最新舱内状态回答。",
        "给我驾驶员、道路和车辆三方面的总览。",
    ):
        topics, _ = selector.detect_topics(text)
        assert ContextTopic.DRIVER in topics
