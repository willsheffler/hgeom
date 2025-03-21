import hgeom

def main():
    test_line_align()

def test_line_align():
    test_code = """
def some_function():

    x = 5
    y = 10

    if x > y: print("x is greater than y")
    else:     print("y is greater than or equal to x")

    excludefile: str = '[gitroot].yapf_exclude',
    hashfile:    str = '[gitroot].yapf_hash',
    conffile:    str = '[gitroot]/pyproject.toml'):

    result = x + y

    a = x.reshape(shape1              + (1, ) * len(shape1) + (4, 4))
    b = y.reshape((1, ) * len(shape2) + shape2              + (4, 4))

    return result
"""
    result = hgeom.identify_formatted_blocks(test_code, 1)
    print(result)

if __name__ == "__main__":
    main()
