import os
import time

from termcolor import colored


class FileChecker:

    def __init__(self, file: str):
        self.filename: str = file.split("/")[-1]
        self.file2check: str = ".last-mod-" + self.filename
        self.file_list: list = []
        self.t_modified: float = 0.0
        self.rebuild_triggered: bool = False
        if file != "":
            self.check_file(file)

    def check_file(self, file: str):
        # print("added watch for file:", file.split("/")[-1])
        self.file_list.append(file)

    def check_rebuild(self):
        # Get stamp of last modification
        with open(self.file2check, "a+") as f:
            f.seek(0)
            data = f.readline()

            # rebuild if empty and insert stamp
            if data == "":
                self.rebuild_triggered = True
                print(colored(self.filename + " was never build on this system, building...", "yellow"))
                self.t_modified = time.time()
                return True

            t_last_build = float(data)

        trigger_rebuild: bool = False
        for file in self.file_list:
            t_modified = os.stat(file).st_mtime
            if t_modified > t_last_build:
                print(colored(self.filename + " was changed, rebuilding...", "yellow"))
                self.t_modified = t_modified
                trigger_rebuild = True

        if trigger_rebuild:
            return True
        return False

    def write_stamp(self):
        with open(self.file2check, "w") as f:
            f.write(str(self.t_modified))
