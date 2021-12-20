#GET CARDS
# Relevant packages
import json, requests, os
from PIL import Image
 
# Search conditions which it uses to go onto the scryfall API
searchCondition = '(set:stx or set:grn or set:war) r:c unique:art -type:land'
#searchCondition = 'serra angel'

# Method which gets data from the API by its search parameters
def getMagicCard(search):
    # Makes a request with the search conditions
    scryfallAPI = requests.get("https://api.scryfall.com/cards/search?q={}".format(search))

    # Checks the status code, to make sure it does not return a faulty database
    if scryfallAPI.status_code == 200:
        # Saves as a JSON file
        scryfallJSON = [scryfallAPI.json()]
        
        # Returns the json file
        return scryfallJSON
    else:
        print("api.scryfall:\n\tstatus_code:", scryfallAPI.status_code)

# Gets the cards from the API using the above method
cards = getMagicCard(searchCondition)

# Path in which the program saves the images to
save_path= 'card_data_base/'

# For loop that renames, resizes and saves every image
for card in cards[0]['data']:
    # Makes sure the cand language is english for whatever reason.
    if card['lang'] == 'en':
        # Gets the image needed
        r = requests.get(card['image_uris']['art_crop'])

        # Names the card
        card_name=str(card['name'] + '.jpg')
        
        # Properly names the files 
        card_name = card_name.replace(" ", "-")
        card_name = card_name.replace(",", "")
        card_name = card_name.replace("'", "")
        card_name = card_name.lower()
        
        #Saves the file in the proper folder with its name
        completeName = os.path.join(save_path, card_name)
        open(completeName, 'wb').write(r.content)

        # Resizes the file by opening it, resizing and resaving
        f_img = completeName
        img = Image.open(f_img)
        img = img.resize((280,190))
        img.save(f_img)