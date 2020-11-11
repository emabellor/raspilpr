from openalpr import Alpr


def main():
    print('Initializing main function')
    
    alpr = Alpr("us", "/usr/local/share/openalpr/config/openalpr.defaults.conf", \
                        "/usr/local/share/openalpr/runtime_data")

    if not alpr.is_loaded():
        print('Error loading alpr')
        sys.exit(1)

    file = '/home/pi/testPlate.jpg'
    results = alpr.recognize_file(file)

    print('Total results: ', len(results['results']))

    print('Unloading ALPR')
    alpr.unload()

    print('Done!')


if __name__ == '__main__':
    main()
