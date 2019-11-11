import sys

ord2chr = ['A', 'B', 'C', 'D',
           'E', 'F', 'G', 'H',
           'I', 'J', 'K', 'L',
           'M', 'N', 'O', 'P',
           'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X',
           'Y', 'Z']

chr2ord = {ord2chr[i]: i for i in range(len(ord2chr))}


def encryption(message, rot):
    """
    Encrypt the message using the rot
    :param message: the message
    :param rot: the rotation value
    :return:
    """
    message = message.upper()
    result = ""
    for i in range(len(message)):
        if message[i].isalpha():
            result += ord2chr[(chr2ord[message[i]] + rot) % 26]
        else:
            result += message[i]
    return result


def decryption(message, rot):
    """
    Decrypt the message using the rot
    :param message: the message
    :param rot: the rotation value
    """
    message = message.upper()
    result = ""
    for i in range(len(message)):
        if message[i].isalpha():
            result += ord2chr[(chr2ord[message[i]] - rot + 26) % 26]
        else:
            result += message[i]
    return result


def get_cipher_mode():
    """
    Get the cipher mode
    """
    while True:
        mod = input("Please input cipher mode(encryption or decryption):")
        if mod == "encryption" or mod == "decryption":
            return mod
        else:
            print("Please choose from encryption or decryption")


def get_rotation_value():
    """
    Get the rotation value
    """
    while True:
        num = input("Please input rotation value:")
        if num.isdigit():
            return int(num)
        else:
            print("Please input a number")


def get_message():
    """
    Get the message to be encrypt or decrypt
    """
    print("Please input messages")
    while True:
        msg = sys.stdin.readlines()
        msg = "".join(msg)
        if len(msg) > 0:
            return msg
        else:
            print("Please input some content")


def get_message_from_file():
    """
    Get the message to be encrypt or decrypt from file
    """
    fn = input("Please input the file name:")
    with open(fn) as fp:
        msg = fp.readlines()
        msg = "".join(msg)
        return msg


def analyse_message(message):
    """
    Analyse the total number of words, unique words,
    min max avg word length,most common letter and
    the most common words
    :param message:
    :return:
    """
    message = message.replace("\n", " ")
    word_count = {}
    letter_count = {}
    word_len = []
    for word in message.split():
        print(word)
        # get word length
        word_len.append(len(word))
        # count the word
        if word not in word_count:
            word_count[word] = 0
        word_count[word] += 1
        # count the letter
        for l in word:
            if l not in letter_count:
                letter_count[l] = 0
            letter_count[l] += 1
    print("Total number of words:" + str(sum(word_count.values())))
    print("Number of unique words:" + str(len(word_count)))
    print("Minimum word length:" + str(min(word_len)))
    print("Maximum word length:" + str(max(word_len)))
    print("Average word length:" + str(sum(word_len) / len(word_len)))
    letter_count_sort = [(i[1], i[0]) for i in letter_count.items()]
    letter_count_sort.sort()
    print("Most common letter:" + str(letter_count_sort[0][1]))
    word_count_sort = [(i[1], i[0]) for i in word_count.items()]
    word_count_sort.sort()
    word_count_sort = word_count_sort[:10]
    print("Ten most common words:")
    for i in word_count_sort:
        print(i[1] + ":" + str(i[0]))


def get_read_choice():
    """
    Get the read choice
    from file or from command
    """
    while True:
        choice = input("Message from file or command(please input file or command):")
        if choice == "file" or choice == "command":
            return choice
        else:
            print("Please choose from file or command")


def get_decryption_mode():
    """
    Get the decryption mode
    :return:
    """
    while True:
        choice = input("Please input decryption mode(hand or auto):")
        if choice == "hand" or choice == "auto":
            return choice
        else:
            print("Please choose from hand or auto")


def auto_decryption(message):
    """
    Automated decryption
    :param message:
    :return:
    """

    words = set()
    # read list
    with open("english dictionary-500.txt") as fp:
        for line in fp.readlines():
            line = line.replace("\n", "")
            words.add(line.upper())
    print(words)
    lines = message.split("\n")
    result = ""
    for line in lines:
        for i in range(26):
            tmp = decryption(line, i)
            flag = False
            for w in tmp.split():
                if w in words:
                    flag = True
                    break
            # match success
            if flag:
                print(tmp)
                ans = input("Is the decryption right?(yes or no):")
                if ans == "yes":
                    print("rotation:" + str(i))
                    result += tmp

    return result


def main():
    """
    The main program
    """
    mode = get_cipher_mode()
    choice = get_read_choice()

    if mode == "encryption":
        rot = get_rotation_value()

    elif mode == "decryption":
        decryption_mode = get_decryption_mode()
        if decryption_mode == "hand":
            rot = get_rotation_value()

    if choice == "file":
        msg = get_message_from_file()
    elif choice == "command":
        msg = get_message()

    if mode == "encryption":
        print(encryption(msg, rot))
        analyse_message(msg)
    elif mode == "decryption":
        if decryption_mode == "hand":
            msg = decryption(msg, rot)
            print(msg)
        elif decryption_mode == "auto":
            msg = auto_decryption(msg)

        analyse_message(msg)


if __name__ == "__main__":
    main()
