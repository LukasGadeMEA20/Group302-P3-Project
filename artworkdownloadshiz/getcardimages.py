#GET CARDS

import json, requests
 
with open(r'C:\Users\mail\OneDrive\Dokumenter\scryfallimgdownload\code shit\unique-artwork-20211209101312.json', 'r', encoding='utf-8') as f:
    cards = json.load(f)
 
for card in cards:
    if card['set'] == 'stx' and card['lang'] == 'en':
        r = requests.get(card['image_uris']['art_crop'])
        card_name=card['name'] + '.jpg'
        open(card_name, 'wb').write(r.content)