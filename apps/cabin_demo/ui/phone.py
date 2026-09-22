import cv2


def open_camera(
    camera_id: int = 0,
):

    cap = cv2.VideoCapture(
        camera_id,
        cv2.CAP_AVFOUNDATION,
    )

    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError("Failed to open camera.")

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    return cap


def draw_interaction_zone(
    frame,
    zone,
):

    if zone is None:
        return

    cv2.rectangle(
        frame,
        (
            zone.x1,
            zone.y1,
        ),
        (
            zone.x2,
            zone.y2,
        ),
        (255, 180, 0),
        2,
    )

    cv2.putText(
        frame,
        "DRIVER INTERACTION ZONE",
        (
            zone.x1,
            max(
                20,
                zone.y1 - 10,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 180, 0),
        1,
        cv2.LINE_AA,
    )
