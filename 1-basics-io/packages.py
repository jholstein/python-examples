#Test for regular packages


import regularpackage1 as rp


import regularpackage1.parent.child.three as three

import namespacepackage1

def main():
    print("hello")
    print(rp.__path__)
    print(three.test3())


if __name__=='__main__':
    main()
