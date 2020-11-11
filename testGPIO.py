from gpiozero import LED
from time import sleep


def main():
    led_1 = LED(20)
    led_2 = LED(21)

    while True:
        led_1.on()
        led_2.on()
        sleep(1)
        led_1.off()
        led_2.off()
        sleep(1)


if __name__ == '__main__':
    main()
