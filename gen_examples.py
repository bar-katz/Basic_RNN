import sys
import random


def generate(max_len, is_good):
    letters = [letter * random.randint(1, max_len) for letter in ['a', 'b', 'c', 'd']]
    numbers = [generate_num(max_len) for i in range(len(letters) + 1)]

    # if generating bad example reorder items
    if not is_good:
        # generate lists of indexes
        letters_reorder = list(range(len(letters)))
        numbers_reorder = list(range(len(letters) + 1))

        # shuffle the indexes
        # shuffle uses an Fisher-Yates shuffle algorithm that ensures
        # perfect shuffle (assuming a good random number generator) thus high probability this loop will take O(1)
        # because shuffle uses a different random each call
        while letters_reorder == list(range(len(letters))):
            random.shuffle(letters_reorder)

        while numbers_reorder == list(range(len(letters) + 1)):
            random.shuffle(numbers_reorder)

        letters = [letters[i] for i in letters_reorder]
        numbers = [numbers[i] for i in numbers_reorder]

    # concat the sequences
    seq = [number + letter for number, letter in zip(numbers, letters)]
    seq.append(numbers[-1])

    return ''.join(seq)


def generate_num(max_len):
    return ''.join([str(random.randint(0, 9)) for i in range(random.randint(1, max_len))])


def main():
    file_name = sys.argv[1]
    num_examples = int(sys.argv[2])
    max_length = int(sys.argv[3])

    pos = [generate(max_length, True) + ' 1' for i in range(int(num_examples / 2))]
    neg = [generate(max_length, False) + ' 0' for i in range(int(num_examples / 2))]
    examples = pos + neg
    random.shuffle(examples)
    examples_str = '\n'.join(examples)

    with open(file_name, 'w') as examples_file:
        examples_file.write(examples_str)


if __name__ == '__main__':
    main()

