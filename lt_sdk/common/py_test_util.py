import cProfile
import getopt
import logging
import multiprocessing
import os
import pstats
import shutil
import sys
import unittest

from lt_sdk.common import py_file_utils

RESULTS_FILE = os.path.join(py_file_utils.TMP_DIR, "results.txt")


class PythonTestCase(unittest.TestCase):

    def setUp(self):
        self._delete_tmp_dir = True
        self.tmp_dir = py_file_utils.mkdtemp()

    def tearDown(self):
        if self._delete_tmp_dir:
            shutil.rmtree(self.tmp_dir)
        else:
            logging.warning("tmp directory not being deleted, location is: {}".format(
                self.tmp_dir))

    def write_test_result(self, result):
        try:
            with open(RESULTS_FILE, "a") as f:
                f.write(result)
        except IOError as e:
            logging.error("Could not open %s: %s" % (RESULTS_FILE, str(e)))


class PythonTestProgram(unittest.TestProgram):

    # Logging constants
    LOGGING_LEVEL = logging.DEBUG
    LOGGING_FORMAT = "%(levelname)s: %(message)s"

    # Colors
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    END = "\033[0m"

    @staticmethod
    def set_root_logger(logging_level=None, logging_format=None):
        # Default values
        if logging_level is None:
            logging_level = PythonTestProgram.LOGGING_LEVEL
        if logging_format is None:
            logging_format = PythonTestProgram.LOGGING_FORMAT

        # Set up root logger
        root = logging.getLogger()
        root.setLevel(logging_level)
        hdlr = logging.StreamHandler(stream=sys.stdout)
        hdlr.setFormatter(logging.Formatter(logging_format))
        hdlr.setLevel(logging_level)
        root.handlers = [hdlr]

    def __init__(self, **kwargs):
        # Default values
        self.set_root_logger()
        self._single_process = False
        self._cprofile_out = None

        if "argv" not in kwargs:
            # Check to see if we should use a single process
            argv = sys.argv
            options, remainder = getopt.gnu_getopt(argv[1:], "s",
                                                   ["single_process", "cprofile_out="])
            for opt, arg in options:
                if opt in ["-s", "--single_process"]:
                    self._single_process = True
                if opt == "--cprofile_out":
                    self._cprofile_out = arg

            # The base class will parse the argv in kwargs, so we need to
            # remove our custom flags before calling super().__init__
            argv = argv[0:1] + remainder
            kwargs["argv"] = argv

        # cProfile can only be used if we are also using a single process
        if not self._single_process and self._cprofile_out is not None:
            raise RuntimeError("cProfile can only be run if using a single process, " +
                               "enable using the -s flag")

        super().__init__(**kwargs)

    @staticmethod
    def _unpack_tests(test_suite):
        unpacked_tests = []
        for test in test_suite:
            if isinstance(test, unittest.suite.TestSuite):
                # Recursively unpack if necessary
                unpacked_tests.extend(PythonTestProgram._unpack_tests(test))
            else:
                # Create a TestSuite for a single test
                single_test = unittest.suite.TestSuite()
                single_test.addTest(test)
                unpacked_tests.append(single_test)

        return unpacked_tests

    @staticmethod
    def _run_single_test(test_runner, test, test_ran, test_passed):
        result = test_runner.run(test)
        test_ran.value = True
        test_passed.value = result.wasSuccessful()

    @staticmethod
    def _colored_string(string, color):
        return color + string + PythonTestProgram.END

    @staticmethod
    def _get_description(test):
        original_description = str(list(test)[0])
        split = original_description.split(" ")
        fn_name = split[0]
        cls_name = split[1]  # format is (__main__.Class)
        cls_name = cls_name[1:-1].split(".")[1]

        return "{}.{}".format(cls_name, fn_name)

    def runTests(self):
        if self._single_process:
            if self._cprofile_out is None:
                # Just use default function
                super().runTests()
            else:
                # Use default function with cProfile and print top 200 results
                cProfile.runctx("super(PythonTestProgram, self).runTests()",
                                globals(),
                                locals(),
                                filename=self._cprofile_out)
                logging.info("Printing stats from cProfile")
                p = pstats.Stats(self._cprofile_out)
                p.sort_stats("cumulative").print_stats(200)
            return

        if self.testRunner is None:
            self.testRunner = unittest.TextTestRunner(verbosity=self.verbosity)

        unpacked_tests = self._unpack_tests(self.test)
        tests_passed = []
        descriptions = []
        for test in unpacked_tests:
            test_ran = multiprocessing.Value("b", False)
            test_passed = multiprocessing.Value("b", False)

            # Run a test in its own process
            p = multiprocessing.Process(target=PythonTestProgram._run_single_test,
                                        args=(self.testRunner,
                                              test,
                                              test_ran,
                                              test_passed))
            p.start()
            p.join()

            if not test_ran.value:
                # Process was terminated before result was added to the queue
                logging.error("Process terminated unexpectedly")
                result = self.testRunner._makeResult()
                result.addError(list(test)[0], sys.exc_info())
                result.printErrors()
                test_passed.value = result.wasSuccessful()

            tests_passed.append(test_passed.value)
            descriptions.append(self._get_description(test))

        # Print summary at the end
        summary = ["", "", "=" * 70, "TEST SUMMARY", "-" * 70]
        for i, d in enumerate(descriptions):
            if tests_passed[i]:
                summary.append(self._colored_string("PASSED: {}".format(d),
                                                    self.OKGREEN))
            else:
                summary.append(self._colored_string("FAILED: {}".format(d), self.FAIL))

        summary.extend(["=" * 70, "", ""])
        summary = "\n".join(summary)

        logging.info(summary)

        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r") as f:
                results_txt = f.read()
            results = ["", "", "=" * 70, "TEST RESULTS", "-" * 70]
            results.extend(results_txt.split("\n"))
            results.extend(["=" * 70, "", ""])
            results = "\n".join(results)
            logging.info(results)
            try:
                os.remove(RESULTS_FILE)
            except IOError as e:
                logging.error("Could not remove %s: %s" % (RESULTS_FILE, str(e)))

        sys.exit(not all(tests_passed))


main = PythonTestProgram
