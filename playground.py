import torch
import atexit


def error_message():
    print('Error message')


def success_message():
    print('Success message')


def main():
    a = torch.rand(1)
    print(a)
    success_message()


if __name__ == '__main__':
    main()
