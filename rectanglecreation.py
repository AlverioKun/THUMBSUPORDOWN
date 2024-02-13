task = []

newTask = input("Add task (A) or change status (CS) or view tasks (V)? ")

while newTask != 0:
    if newTask == "A":
        taskDescription = input("What's the task? ")
        task.append(taskDescription)
        newTask = input("Add task (A) or change status (CS) or view tasks (V)? ")

    elif newTask == "CS":
        completedTaskNumber = input("Which task is completed?")
        completedTaskNumber = int(completedTaskNumber)
        task.pop(completedTaskNumber)
        newTask = input("Add task (A) or change status (CS) or view tasks (V)? ")

    elif newTask == "V":
            print(task)

    elif newTask == "End Task":
         newTask = 0
    
    else:
         print("Enter a valid input")






