x = 0
myList = []
while x != "esc":
    x = input("Please insert list: ")
    if x.lower() ==  "esc":
        break
    else:
        myList.append(x)
myList.reverse()
print(myList)

myList = list(input("insert list: ").split(" "))
print(myList)