#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

from tqdm import tqdm
import numpy as np


def read_pool(filepath):
    print(f'Read {filepath}')
    with open(filepath) as f:
        data = [json.loads(line) for line in tqdm(f)]
    return set([(record['task'], record['list']) for record in tqdm(data)])


GENERAL_LIST_NAMES = ['to do', 'reminders']
def generate_examples(names, verb, listname, label,
                      exclude_pairs=set(), exclude_names=set(), n=None):
    results = []
    valid = set()
    for name in names:
        name = name.lower()
        if isinstance(n, int) and len(results) >= n:
            break
        if name in exclude_names:
            continue
        buff = []
        for glistname in GENERAL_LIST_NAMES:
            if (f'{verb} {name}', glistname) in exclude_pairs:
                continue
            buff.append((f'{verb} {name}', glistname, f'({verb} <{label}>, {glistname})'))
        if (name, listname) in exclude_pairs:
            continue
        buff.append((name, listname, f'(<{label}>, {listname})'))
        if len(buff) > 1:
            results += buff
            valid.add(name)
    print('Found', len(results), 'for', label)
    print(valid)
    return results


def main(args):
    pool = set()
    for path_pool in args.path_pool:
        pool |= read_pool(path_pool)
    print(f'Read {len(pool)} task-list pairs')

    # https://www.ssa.gov/oact/babynames/decades/century.html
    NAMES = ['James', 'Robert', 'John', 'Michael', 'William', 'David', 'Richard', 'Joseph', 'Thomas', 'Charles', 'Christopher', 'Daniel', 'Matthew', 'Anthony', 'Mark', 'Donald', 'Steven', 'Paul', 'Andrew', 'Joshua', 'Kenneth', 'Kevin', 'Brian', 'George', 'Edward', 'Ronald', 'Timothy', 'Jason', 'Jeffrey', 'Ryan', 'Jacob', 'Gary', 'Nicholas', 'Eric', 'Jonathan', 'Stephen', 'Larry', 'Justin', 'Scott', 'Brandon', 'Benjamin', 'Samuel', 'Gregory', 'Frank', 'Alexander', 'Raymond', 'Patrick', 'Jack', 'Dennis', 'Jerry', 'Tyler', 'Aaron', 'Jose', 'Adam', 'Henry', 'Nathan', 'Douglas', 'Zachary', 'Peter', 'Kyle', 'Walter', 'Ethan', 'Jeremy', 'Harold', 'Keith', 'Christian', 'Roger', 'Noah', 'Gerald', 'Carl', 'Terry', 'Sean', 'Austin', 'Arthur', 'Lawrence', 'Jesse', 'Dylan', 'Bryan', 'Joe', 'Jordan', 'Billy', 'Bruce', 'Albert', 'Willie', 'Gabriel', 'Logan', 'Alan', 'Juan', 'Wayne', 'Roy', 'Ralph', 'Randy', 'Eugene', 'Vincent', 'Russell', 'Elijah', 'Louis', 'Bobby', 'Philip', 'Johnny', 'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen', 'Nancy', 'Lisa', 'Betty', 'Margaret', 'Sandra', 'Ashley', 'Kimberly', 'Emily', 'Donna', 'Michelle', 'Dorothy', 'Carol', 'Amanda', 'Melissa', 'Deborah', 'Stephanie', 'Rebecca', 'Sharon', 'Laura', 'Cynthia', 'Kathleen', 'Amy', 'Shirley', 'Angela', 'Helen', 'Anna', 'Brenda', 'Pamela', 'Nicole', 'Emma', 'Samantha', 'Katherine', 'Christine', 'Debra', 'Rachel', 'Catherine', 'Carolyn', 'Janet', 'Ruth', 'Maria', 'Heather', 'Diane', 'Virginia', 'Julie', 'Joyce', 'Victoria', 'Olivia', 'Kelly', 'Christina', 'Lauren', 'Joan', 'Evelyn', 'Judith', 'Megan', 'Cheryl', 'Andrea', 'Hannah', 'Martha', 'Jacqueline', 'Frances', 'Gloria', 'Ann', 'Teresa', 'Kathryn', 'Sara', 'Janice', 'Jean', 'Alice', 'Madison', 'Doris', 'Abigail', 'Julia', 'Judy', 'Grace', 'Denise', 'Amber', 'Marilyn', 'Beverly', 'Danielle', 'Theresa', 'Sophia', 'Marie', 'Diana', 'Brittany', 'Natalie', 'Isabella', 'Charlotte', 'Rose', 'Alexis', 'Kayla', 'Mia', 'Kinsley', 'Ezekiel', 'Liam', 'Noah', 'Oliver', 'Elijah', 'William', 'James', 'Benjamin', 'Lucas', 'Henry', 'Alexander', 'Olivia', 'Emma', 'Ava', 'Charlotte', 'Sophia', 'Amelia', 'Isabella', 'Mia', 'Evelyn', 'Harper']
    NAMES = sorted(list(NAMES))
    print(len(NAMES), 'names')

    # https://vegetablesfruitsgrains.com/
    np.random.seed(42)
    GROCERIES = ['Acorn', 'Adzuki Beans', 'Amaranth', 'Amaranth Leaves', 'Anasazi Beans', 'Apple', 'Apricot', 'Arborio', 'Artichoke', 'Arugula', 'Asian Pear', 'Asparagus', 'Aubergine', 'Avocado', 'Baby Corn', 'Bamboo Shoots', 'Banana', 'Barley', 'Barley Flakes', 'Barley Grits', 'Barley Groats', 'Basmati', 'Bean Sprouts', 'Beet', 'Beet Greens', 'Belgian Endive', 'Bell Pepper', 'Bengal Gram', 'Bhutanese Red', 'Bitter Gourd', 'Bitter Melon', 'Black Beans', 'Black Forbidden Rice', 'Black Japonica', 'Black Turtle Beans', 'Black-Eyed Peas', 'Blood Orange', 'Bok Choi', 'Bok Choy', 'Borlotti', 'Breadfruit', 'Broad Beans', 'Broccoli', 'Brown', 'Brussels Sprouts', 'Buckwheat', 'Buckwheat Grits', 'Buckwheat Groats', 'Bulgur Wheat', 'Burdock Root', 'Butter Beans', 'Buttercup', 'Butterhead- Bibb, Boston', 'Butternut', 'Cabbage', 'Cactus Pear', 'Calabash', 'Calasparra', 'Calrose', 'Camargue Red Rice', 'Candle Corn', 'Cannellini Beans', 'Capers', 'Carnaroli', 'Carob', 'Carrot', 'Cassava', 'Cauliflower', 'Celeriac', 'Celery', 'Celery Root', 'Celtuce', 'Chana Dal', 'Chayote', 'Cherry', 'Chestnut Lima', 'Chickpeas', 'Chinese Broccoli', 'Chinese Spinach', 'Christmas Lima Bean', 'Citron', 'Coconut', 'Collard Greens', 'Converted', 'Corn', 'Corona Beans', 'Couscous', 'Cracked Rye', 'Cracked Wheat', 'Cranberry', 'Cucumber', 'Curly', 'Currant', 'Cushaw', 'Daikon Radish', 'Damson', 'Dandelion Greens', 'Date', 'Delicata', 'Desi', 'Dragon Fruit', 'Durian', 'Durum Wheat', 'Edamame', 'Eggplant', 'Elephant Garlic', 'Endive', 'English Cucumber', 'Escarole', 'Farina', 'Farro', 'Fava Beans', 'Fennel', 'Fiddlehead', 'Fig', 'Flageolet Beans', 'Forbidden', 'Frisee', 'Galangal', 'Garbanzo Beans', 'Garlic', 'Gherkin', 'Ginger', 'Glutinous', 'Gobo', 'Grape Leaves', 'Grapefruit', 'Grapes', 'Great Northern Beans', 'Green Beans', 'Green Onions', 'Green, Red, Yellow, Brown, Other', 'Green, Yellow', 'Greens', 'Guava', 'Hearts of Palm', 'Hominy', 'Horseradish', 'Hubbard', 'Hulled Barley', 'Iceberg', 'Instant', 'Instant Oatmeal', 'Instant Oats', 'Irish Oatmeal', 'Jackfruit', 'Jasmine', 'Jerusalem Artichoke', 'Jujube', 'Jícama', 'Kabocha', 'Kai-lan', 'Kala Jeera', 'Kale', 'Kalijira', 'Kamut', 'Kasha', 'Kidney Beans', 'Kiwifruit', 'Kohlrabi', 'Kohlrabi Greens', 'Kumquat', 'Lacinato', 'Leaf- Green Leaf, Red Leaf', 'Leeks', 'Lemon', 'Lemongrass', 'Lentils', 'Lettuce', 'Lima Beans', 'Lime', 'Long Grain', 'Longan', 'Loquat', 'Lotus Root', 'Lotus Seed', 'Lucuma', 'Lupini Beans', 'Lychee', 'Maize', 'Mamey Sapote', 'Mandarin- Clementine, Satsuma', 'Mango', 'Mangosteen', 'Marrow Beans', 'Medium Grain', 'Millet', 'Mochi', 'Moth Beans', 'Mung Beans', 'Mustard Greens', 'Nance', 'Napa Cabbage', 'Navel', 'Navy Beans', 'Nectarine', 'Noni', 'Nopales', 'Oat Bran', 'Oat Groats', 'Oats', 'Okra', 'Old Fashioned Oatmeal', 'Olive', 'Onion', 'Oranges', 'Ornamental', 'Paella', 'Pak Choy', 'Papaya', 'Parboiled', 'Parsley', 'Parsley Root', 'Parsnip', 'Passion Fruit', 'Pasta', 'Patna', 'Patty Pan', 'Peach', 'Pear', 'Pearl', 'Pearled Barley', 'Peas', 'Persimmon', 'Pickling Cucumbers', 'Pigeon Pea', 'Pineapple', 'Pink Beans', 'Pinto Beans', 'Pitaya', 'Plantain', 'Plum', 'Pomegranate', 'Pomelo', 'Popcorn', 'Popcorn Rice', 'Pot', 'Potato', 'Prickly Pear', 'Prunes', 'Pumpkin', 'Purple Sticky', 'Purslane', 'Quick Cooking Oats', 'Quick Oatmeal', 'Quince', 'Quinoa', 'Radicchio', 'Radish', 'Raisin', 'Rambutan', 'Rapini', 'Red Beans', 'Rhubarb', 'Rice Beans', 'Riz Rouge', 'Roasted Buckwheat', 'Rolled Oats', 'Romaine', 'Roman', 'Rutabaga', 'Rye', 'Rye Berries', 'Rye Flakes', 'Scallions', 'Scarlet Runner Beans', 'Scotch Barley', 'Seitan', 'Semolina', 'Seville', 'Shallots', 'Short Grain', 'Small Red Beans', 'Snap Beans', 'Sorghum', 'Soya Bean', 'Soybean', 'Spaghetti', 'Spanish Tolosana Beans', 'Spelt', 'Spelt Berries', 'Spelt Flakes', 'Spesinach', 'Split Peas', 'Starfruit', 'Steel cut oats', 'Sticky', 'String Beans', 'Summer Squash', 'Sunchokes', 'Sushi', 'Sweet', 'Sweet Potato', 'Swiss Chard', 'Tamarillo', 'Tamarind', 'Tangelo', 'Tangerine', 'Taro', 'Teff', 'Tepary Beans', 'Tomatillo', 'Tomato', 'Triticale', 'Triticale Berries', 'Triticale Flakes', 'Turban', 'Turnip', 'Turnip Greens', 'Unroasted Buckwheat', 'Urad', 'Valencia', 'Water Chestnut', 'Water Spinach', 'Watercress', 'Wax Beans', 'Wheat', 'Wheat Berries', 'Wheat Bran', 'Wheat Flakes', 'Wheat Gluten', 'White', 'Wild Rice', 'Winter Melon', 'Winter Squash', 'Yams', 'Yellow- Straightneck, Crookneck', 'Yuca', 'Zucchini', 'green peas', 'snow peas', 'sugar snap peas']
    GROCERIES = sorted(list(GROCERIES))
    np.random.shuffle(GROCERIES)
    print(len(GROCERIES), 'grocery items')
    

    results_name = generate_examples(NAMES, 'call', 'to call', 'person',
                                     exclude_pairs=pool)

    results_grocery = generate_examples(GROCERIES, 'buy', 'to buy', 'grocery',
                                        exclude_pairs=pool,
                                        exclude_names={'white', 'sweet', 'yellow', 'glutinous',
                                                       'ornamental', 'converted'},
                                        n=len(results_name))

    print('Write the results to', args.path_output)
    with open(args.path_output, 'w') as f:
        for (taskname, listname, label) in results_name + results_grocery:
            f.write(f'{taskname}\t{listname}\t{label}\n')

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', dest='path_pool', nargs='+', default=['data/v1/trn.json', 'data/v1/tst.json'])  # to make sure that the generated toy examples do not exist in the Wunderlist dataset
    parser.add_argument('-o', '--output', dest='path_output', default='data/buy_groceries_vs_call_persons/v1.tsv', help='path to an output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()
    main(args)
