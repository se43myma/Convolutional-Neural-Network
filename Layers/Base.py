#base layer inherited by other layers
class BaseLayer:
    def __init__(self) -> None:
        self.trainable = False  #default network is not trainable
