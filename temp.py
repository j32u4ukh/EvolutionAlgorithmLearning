class Test:
    def __init__(self):
        self.name = "test"

    def private(self):
        if __name__ == '__main__':
            print(self.name)
        else:
            print("This is private function")

    def public(self):
        self.private()


t = Test()
t.private()
t.public()
