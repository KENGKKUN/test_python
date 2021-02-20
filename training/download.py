import pyrebase


config = {
    'apiKey': "AIzaSyAQKROF-qLGbcQ4Dx8gyME2byQkis2wPNw",
    'authDomain': "durian-classification.firebaseapp.com",
    'databaseURL': "https://durian-classification.firebaseio.com",
    'storageBucket': "durian-classification.appspot.com",
    'serviceAccount': "durian-classification-firebase-adminsdk-5eu3x-d4aac22f56.json"
}

firebase = pyrebase.initialize_app(config)


storage = firebase.storage()


def download_data(file_in_db, path, file_name):
    storage.child(f'Audio/{file_in_db}').download(path, f'{file_name}')
    print('Data Downloaded!')


if __name__ == '__main__':
    all_files = storage.child('Audio1').list_files()

    datadir = 'sounds/unripe_day3/'



    # print(all_files)

    for file in all_files:
        # print(file.name.split("/")[1])
        try:
            file.download_to_filename(datadir + file.name.split("/")[1])
        except:
            print('Download Failed')


