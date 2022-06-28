from vertex import Vertex


class ROI:

    def __init__(self, width, height) -> None:
        self.corners = [Vertex(0, 0), Vertex(width - 1, 0), Vertex(width - 1, height - 1), Vertex(0, height - 1)]

    