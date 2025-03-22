import unittest
import hgeom

def main():
    # print(hgeom.OTHER)
    obj = hgeom.IdentifyFormattedBlocks()
    print(obj)

class TestIdentifyFormattedBlocks(unittest.TestCase):

    def setUp(self):
        # Create an instance of the class
        self.obj = hgeom.IdentifyFormattedBlocks()
        # self.NUM_GROUPS = hgeom.NUM_GROUPS

    def test_set_substitution_matrix(self):
        # This test simply ensures that the setter does not raise errors.
        try:
            self.obj.set_substitution_matrix(hgeom.CharGroup.EQUAL, hgeom.CharGroup.COLON, 2.5)
        except Exception as e:
            self.fail(f"set_substitution_matrix raised an exception: {e}")

    def test_compute_similarity_score_identical_lines(self):
        score1 = self.obj.compute_similarity_score("hello", "hello")
        score2 = self.obj.compute_similarity_score("[]()", "[]()")
        assert score1 < score2

    def test_strange_cases(self):
        assert self.obj.mark_formtted_blocks('', threshold=0.99) == ''
        assert self.obj.mark_formtted_blocks('\n', threshold=0.99) == ''
        assert self.obj.mark_formtted_blocks('\n\n', threshold=0.99) == '\n'
        assert self.obj.mark_formtted_blocks('========', threshold=0.99)
        assert self.obj.mark_formtted_blocks('========\n========', threshold=0.99)
        assert self.obj.mark_formtted_blocks('========\n========\n', threshold=0.99)
        assert self.obj.unmark('') == ''
        assert self.obj.unmark('\n') == ''
        assert self.obj.unmark('\n\n') == ''
        assert self.obj.unmark('========')
        assert self.obj.unmark('========\n========')
        assert self.obj.unmark('========\n========\n')

    def test_unmark(self):
        test = """q
        #             fmt: on
z
        #             fmt: off
a
    #             fmt: off
b
#             fmt: on
c
        """
        test2 = """q
z
a
b
c
        """
        assert self.obj.unmark(test) == test2

    def test_mark_formtted_blocks_no_change(self):
        # If lines are dissimilar (or threshold is set high),
        # the mark_formtted_blocks function should return code without formatting markers.
        code = "line one\nline two"
        result = self.obj.mark_formtted_blocks(code, threshold=5)
        # We strip both strings to avoid any trailing newline differences.
        self.assertEqual(result.strip(), code.strip())

    def test_mark_formtted_blocks_formatting(self):
        # When lines are similar, formatting markers should be inserted.
        # Here we use a low threshold to force detection.
        code = "\n    int a = 0;\n    int a = 0;"
        print(self.obj.compute_similarity_score('int a = 0', 'int a = 0'))
        result = self.obj.mark_formtted_blocks(code, threshold=2)
        self.assertIn("#             fmt: off", result)
        self.assertIn("#             fmt: on", result)

    def test_whitespace(self):
        # Test that the function handles leading/trailing whitespace correctly.
        code = "    line one\n\n\n\n    line two"
        result = self.obj.unmark(code)
        assert len(result.split('\n')) == 3

    def test_inline_blocks_are_marked(self):
        # Test that inline blocks are marked correctly.
        code = """
        def example_function(): foo
            if True: False
                class Banana: ...
        elif bar: baz
            else: qux
        """
        result = self.obj.mark_formtted_blocks(code, threshold=2)
        assert result == """
        #             fmt: off
        def example_function(): foo
        #             fmt: on
            #             fmt: off
            if True: False
            #             fmt: on
                #             fmt: off
                class Banana: ...
                #             fmt: on
        #             fmt: off
        elif bar: baz
        #             fmt: on
            #             fmt: off
            else: qux
            #             fmt: on
        """

    def test_inline_blocks_are_marked2(self):
        # Test that inline blocks are marked correctly.
        code = """
        print('foo')

        def example_function(): foo
            dummy
            if True: False

                dummy2
                class Banana: ...
        elif bar: baz
            else: qux
        aaaa
        """
        result = self.obj.mark_formtted_blocks(code, threshold=2)
        assert result == """
        print('foo')

        #             fmt: off
        def example_function(): foo
        #             fmt: on
            dummy
            #             fmt: off
            if True: False
            #             fmt: on

                dummy2
                #             fmt: off
                class Banana: ...
                #             fmt: on
        #             fmt: off
        elif bar: baz
        #             fmt: on
            #             fmt: off
            else: qux
            #             fmt: on
        aaaa
        """

    def test_multiline_is_ignored(self):
        # Test that inline blocks are marked correctly.
        code = """
        if a: return b \\
        else: return c
        for a in b: c
        """
        result = self.obj.mark_formtted_blocks(code, threshold=4)
        assert result == """
        if a: return b \\
        else: return c
        #             fmt: off
        for a in b: c
        #             fmt: on
        """

if __name__ == "__main__":
    main()
