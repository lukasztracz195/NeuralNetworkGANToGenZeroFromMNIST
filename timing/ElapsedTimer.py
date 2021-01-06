import timing


class ElapsedTimer(object):
    def __init__(self):
        self.__start_time = timing.time()

    @staticmethod
    def __elapsed(sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s " % self.__elapsed(timing.time() - self.__start_time))
