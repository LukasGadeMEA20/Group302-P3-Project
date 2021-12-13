#GET CARDS

import json, requests, os
from PIL import Image
 
#with open(r'C:\Users\mail\OneDrive\Dokumenter\scryfallimgdownload\code shit\unique-artwork-20211209101312.json', 'r', encoding='utf-8') as f:
#    cards = json.load(f)
 
searchCondition = '(set:stx or set:grn or set:war) r:c unique:art -type:land'
#searchCondition = 'serra angel'

def getMagicCard(card):
    scryfallAPI = requests.get("https://api.scryfall.com/cards/search?q={}".format(card))
    if scryfallAPI.status_code == 200:
        #Gemmer den som sin egen JSON fil
        scryfallJSON = [scryfallAPI.json()]
        #Tilføjer navnet fra det element vi er nået til i bibliotekket og bagefter et element fra det data vi får fra api'en.
        #url = scryfallJSON['data'][0]['image_uris']['border_crop']
        #url = scryfallJSON['']
        return scryfallJSON
    else:
        print("api.scryfall:\n\tstatus_code:", scryfallAPI.status_code)

cards = getMagicCard(searchCondition)
save_path= 'card_data_base/'

for card in cards[0]['data']:
    if card['lang'] == 'en':
        r = requests.get(card['image_uris']['art_crop'])
        card_name=str(card['name'] + '.jpg')
        
        card_name = card_name.replace(" ", "-")
        card_name = card_name.replace(",", "")
        card_name = card_name.replace("'", "")
        
        card_name = card_name.lower()
        
        completeName = os.path.join(save_path, card_name)
        
        open(completeName, 'wb').write(r.content)

        f_img = completeName
        img = Image.open(f_img)
        img = img.resize((280,190))
        img.save(f_img)