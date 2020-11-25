
import shutil
import wget


data = { 
        'million_song': 'http://labrosa.ee.columbia.edu/millionsong/sites/default/files/challenge/train_triplets.txt.zip',
        'million_song_meta': 'http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/track_metadata.db'
}


def download_million_song_data():

    print('[+] Fetching Data...')
    wget.download(data['million_song'])
    wget.download(data['million_song_meta'])
    print('[i] Song Data retireved...')

    return None

def retrieve_dataset():

    download_million_song_data()
    print('[+] Unzipping Download...')
    shutil.unpack_archive('train_triplets.txt.zip', 'song_data')
    print('[i] Finished Unzipping...')

    return None


if __name__ == '__main__':

    retrieve_dataset()