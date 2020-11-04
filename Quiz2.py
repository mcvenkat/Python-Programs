import string

NUMBER_OF_ATTEMPTS = 2
ENTER_ANSWER = 'Hit %s for your answer\n'
TRY_AGAIN = 'Incorrect!!! Try again.'
NO_MORE_ATTEMPTS = 'Incorrect!!! You ran out of your attempts'

def question(message, options, correct, attempts=NUMBER_OF_ATTEMPTS):
    '''
    message - string
    options - list
    correct - int (Index of list which holds the correct answer)
    attempts - int
    '''
    optionLetters = string.ascii_lowercase[:len(options)]
    print message
    print ' '.join('%s: %s' % (letter, answer) for letter, answer in zip(optionLetters, options))
    while attempts > 0:
        response = input(ENTER_ANSWER % ', '.join(optionLetters)) # For python 3
        #response = raw_input(ENTER_ANSWER % ', '.join(optionLetters)) # For python 2
        if response == optionLetters[correct]:
            return True
        else:
            attempts -= 1
            print TRY_AGAIN

    print NO_MORE_ATTEMPTS
    return False


print("Mathematics Quiz")

# question1 and question2 will be 'True' or 'False'
question1 = question('Who is president of USA?', ['myself', 'His Dad', 'His Mom', 'Barack Obama'], 3)
question2 = question('Who invented Facebook?', ['Me', 'His Dad', 'Mark Zuckerberg', 'Aliens', 'Someone else'], 2)
