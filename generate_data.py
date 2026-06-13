import random
import json
import re

random.seed(42)

SEED_TYPES = [
    "sunflower seeds", "millet", "cracked corn", "wheat grains", "rice grains",
    "bajra", "ragi", "jowar", "chana dal", "flaxseeds",
    "sesame seeds", "mustard seeds", "pumpkin seeds", "niger seeds", "poppy seeds",
    "quinoa", "amaranth", "barley grains", "oats", "rye grains",
    "safflower seeds", "hemp seeds", "chia seeds", "fenugreek seeds", "fennel seeds",
    "cumin seeds", "caraway seeds", "dill seeds", "anise seeds", "sorghum",
]

PERCH_SPOTS = [
    "water tank edge", "satellite dish", "ac unit", "clothesline", "parapet wall",
    "window sill", "gutter pipe", "roof overhang", "temple spire", "flagpole",
    "tree branch", "balcony rail", "power line", "drain pipe", "terrace ledge",
    "solar panel", "chimney top", "water pipe bend", "awning frame", "shop sign",
    "lamppost", "scaffolding pole", "bus stop roof", "temple dome", "courtyard wall",
    "park bench", "fountain edge", "staircase rail", "veranda beam", "kite string",
]

CITY_SOUNDS = [
    "honking rickshaw", "chai clinking", "temple bells", "traffic hum",
    "construction drill", "kids playing", "vendor calling", "train rumble",
    "music from shop", "auto engine", "siren wailing", "dog barking",
    "people laughing", "plate clattering", "prayer call",
    "bicycle bell", "truck reversing", "street sweeper", "kite flapping", "cricket chirping",
    "water pump", "generator hum", "firecracker pop", "drum beat", "flute playing",
    "baby crying", "tea kettle whistle", "newspaper rustling", "chai walla calling", "auto backfiring",
]

THREATS = [
    "black kite", "street cat", "stray dog", "monkey", "eagle",
    "hawk", "cobra", "mongoose", "crow gang", "lizard",
    "rat", "snake", "owl", "fox", "jackal",
    "prowling cat", "hissing cat", "stalking cat", "big dog", "electric wire",
    "thorn bush", "broken glass", "loud truck", "firework", "fire",
    "feral cat", "golden eagle", "rat snake", "city fox", "stray bull"
]

BIRD_FRIENDS = [
    "crow", "sparrow", "myna", "parrot", "bulbul",
    "koel", "tailor bird", "sunbird", "woodpecker", "drongo",
    "babbler", "lapwing", "kingfisher", "bee eater", "robin",
    "wagtail", "finch", "swallow", "swift", "barbet",
    "coucal", "hornbill", "cuckoo", "heron", "egret",
    "peacock", "owl", "flamingo", "weaver bird", "pelican"
]

CITY_ANIMALS = [
    "squirrel", "chipmunk", "stray cow", "goat", "street pig", "chameleon",
    "street dog", "street cat", "mongoose", "rat"
]

HUMAN_TYPES = [
    "shopkeeper", "little boy", "old lady", "school girl", "traffic cop",
    "sweeper", "postman", "pedestrian", "seed seller", "chai walla"
]

WEATHER_WORDS = [
    "hot sun", "monsoon rain", "humid wind", "foggy mist", "cool breeze",
    "scorching heat", "thundercloud", "lightning flash", "rainbow arc",
    "dust storm", "clear sky", "chilly morning", "gentle drizzle", "heavy downpour",
    "harsh sunlight", "icy wind", "warm breeze", "sticky humidity", "dry spell", "overcast sky",
    "blazing sun", "soft rain", "howling wind", "misty dawn", "golden sunset",
]


def pick(seq):
    return random.choice(seq)


def clean_response(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s\.,\'\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    parts = [s.strip() for s in text.split(".") if s.strip()]
    if not parts:
        return "seeds."
    parts = parts[:3]
    return ". ".join(parts) + "."


FLUFF = [
    "coo", "seeds", "yes coo", "i think so", "for sure",
    "maybe", "coo coo", "i say", "indeed", "truly",
    "hmm yes", "so it goes", "that is all", "coo to you",
    "life is seeds", "yes indeed", "and so", "i see",
    "yes", "right", "ok coo", "seeds yes",
    "good", "alright", "just so", "fair enough", "quite",
    "naturally", "surely", "of course", "yes yes", "indeed coo",
]


# Topic-specific entity lists to prevent semantic leakage
CAT_THREATS = ["street cat", "stray cat", "prowling cat", "hissing cat", "stalking cat", "ginger cat", "black cat"]
SKY_THREATS = ["black kite", "eagle", "hawk", "falcon", "large bird of prey"]
DOG_THREATS = ["stray dog", "street dog", "big dog", "barking dog", "guard dog"]
MONKEY_THREATS = ["monkey", "gray langur", "macaque", "baby monkey", "troop of monkeys"]
SNAKE_THREATS = ["cobra", "snake", "viper", "rat snake", "garden snake"]

CROW_NAMES = ["crow", "house crow", "jungle crow", "old crow", "clever crow"]
SPARROW_NAMES = ["sparrow", "house sparrow", "little sparrow", "chirpy sparrow"]
MYNA_NAMES = ["myna", "common myna", "jungle myna", "chatty myna"]
PARROT_NAMES = ["parrot", "green parrot", "rose-ringed parakeet", "noisy parrot"]


def add_fluff(response):
    raw = response.rstrip(".")
    sentences = [s.strip() for s in raw.split(". ") if s.strip()]
    if len(sentences) == 0:
        return "seeds."
    if len(sentences) >= 3:
        return ". ".join(sentences) + "."
    if len(sentences) == 1 and random.random() < 0.3:
        sentences.append(pick(FLUFF))
    elif len(sentences) == 2 and random.random() < 0.15:
        sentences.append(pick(FLUFF))
    return ". ".join(sentences) + "."


QUESTIONS = {
    "food_found": [
        "what did you eat today", "did you find any seeds", "are you hungry",
        "what is your favorite food", "is there food on the rooftop",
        "do the humans feed you", "where do you find food",
        "what was for breakfast",
    ],
    "food_searching": [
        "what are you looking for", "are there any good spots to eat",
        "where do you search for food", "is it hard to find food",
        "do you ever go hungry", "how do you find food",
        "where are you going", "what are you pecking at",
    ],
    "food_bread": [
        "did you find any bread today", "is there good bread around here",
        "do you like bread", "what kind of bread do you eat",
        "is the bread fresh today", "do you share bread with other birds",
        "i saw bread on the street", "did you eat the roti",
    ],
    "food_rice": [
        "did you eat rice today", "are there rice grains on the roof",
        "do you like rice", "is the rice from the temple",
        "who dropped the rice", "is rice better than seeds",
        "i saw rice near the drain", "do you eat basmati",
    ],
    "food_trash": [
        "what is in the trash today", "do you eat from the garbage",
        "is the trash full of food", "do the humans waste good food",
        "i saw you near the bin", "is there anything good in the dumpster",
        "do you dig through trash", "what did you find in the garbage",
    ],
    "food_seed_preference": [
        "what is your favorite seed", "do you like sunflower seeds",
        "what seeds taste best", "are some seeds better than others",
        "what kind of seeds do you prefer", "do you like millet",
        "which seeds are the tastiest", "do you eat all seeds or just some",
    ],
    "food_water": [
        "where do you find water", "are you thirsty",
        "is the water clean on the roof", "do you drink from puddles",
        "where do birds get water", "is there water on the rooftop",
        "do you share water with other birds", "i saw you drinking from the pipe",
    ],
    "rain_shelter": [
        "where do you go when it rains", "are you getting wet",
        "do you have a shelter", "is the rain too heavy",
        "where do birds hide from rain", "do you like rain",
        "are you warm enough in the rain", "does the roof leak",
    ],
    "rain_enjoy": [
        "do you like the rain", "is the rain nice today",
        "are you bathing in the rain", "do you enjoy wet weather",
        "is the rain refreshing", "do you dance in the rain",
        "are you having fun in the rain", "do you sing in the rain",
    ],
    "rain_wet": [
        "are you wet", "did you get caught in the rain",
        "is your nest wet", "are your feathers dry",
        "do you hate being wet", "did you get soaked",
        "are you shivering", "do you need to dry off",
    ],
    "rain_puddle": [
        "are you drinking from the puddle", "is the puddle water fresh",
        "do you bathe in puddles", "are there good puddles on the roof",
        "do you share puddles with other birds", "is the puddle cold",
        "do you drink from the street puddles", "are puddles better than tap water",
    ],
    "weather_hot": [
        "is it too hot today", "do you like hot weather",
        "where do you go when it is hot", "do you get sunburned",
        "is the roof too hot", "do you hide from the sun",
        "is it hotter than yesterday", "do you pant in the heat",
    ],
    "weather_cold": [
        "is it cold up there", "do you feel cold",
        "do you like winter", "are you shivering",
        "is your nest warm enough", "do the mornings feel chilly",
        "do you huddle with other birds", "is the wind cold at night",
    ],
    "storm_fear": [
        "are you scared of the storm", "do you hear the thunder",
        "is the lightning scary", "do storms damage your nest",
        "do you hide during storms", "are you ok in the storm",
        "do the clouds look dark", "do you sense the storm coming",
    ],
    "threat_cat_spotted": [
        "did you see a cat", "are there cats on the roof",
        "do cats bother you", "are you afraid of cats",
        "did you see the orange cat", "do cats try to catch you",
        "are you careful around cats", "did you see a cat nearby",
    ],
    "threat_cat_chase": [
        "are you running from a cat", "did a cat chase you",
        "are you ok", "did you escape the cat",
        "was it close", "are cats fast",
        "did the cat almost get you", "are you safe now",
    ],
    "threat_cat_escape": [
        "did you get away", "how did you escape",
        "are you safe now", "did the cat give up",
        "where is the cat now", "did you fly to safety",
        "was the cat angry", "do you feel safe now",
    ],
    "threat_kite": [
        "did you see a kite", "are kites dangerous",
        "do you watch for birds of prey", "is there a kite circling",
        "are you afraid of large birds", "do kites eat pigeons",
        "did you see a hawk", "do you hide when kites come",
    ],
    "threat_dog": [
        "are there dogs on the street", "do dogs chase you",
        "did you see a dog", "are dogs dangerous",
        "do dogs bark at you", "did the dog try to bite",
        "are you scared of dogs", "do you stay away from dogs",
    ],
    "threat_monkey": [
        "are there monkeys here", "do monkeys steal food",
        "are monkeys dangerous", "did you see a monkey",
        "do monkeys climb on the roof", "are monkeys scary",
        "did a monkey chase you", "do you share food with monkeys",
    ],
    "threat_snake": [
        "did you see a snake", "are there snakes on the roof",
        "are snakes dangerous", "do snakes eat eggs",
        "did you see a cobra", "are you careful around snakes",
        "do snakes climb to nests", "are there snakes in the city",
    ],
    "flight_soaring": [
        "do you like flying high", "how high can you fly",
        "do you soar above the city", "what is it like up there",
        "do you feel free when flying", "can you see everything from up there",
        "do you fly above the clouds", "is the view nice from up high",
    ],
    "flight_short": [
        "do you fly far", "where do you fly each day",
        "do you prefer short flights", "do you stay close to your rooftop",
        "how far do you travel", "do you fly to other buildings",
        "do you fly across the street", "do you visit other rooftops",
    ],
    "flight_tired": [
        "are you tired", "did you fly a lot today",
        "do your wings hurt", "do you need to rest",
        "was it a long flight", "do you need a nap",
        "are your wings heavy", "should you rest for a while",
    ],
    "flight_wind": [
        "is the wind strong today", "do you like flying in wind",
        "does wind make flying harder", "do you get pushed by the wind",
        "is it fun to ride the wind", "do you glide on the breeze",
        "does the wind help you fly", "are you blown off course",
    ],
    "flight_gliding": [
        "do you like to glide", "how do you glide so smoothly",
        "do you glide between buildings", "is gliding better than flapping",
        "do you enjoy floating on wind", "do you glide to save energy",
        "where do you glide to", "is gliding effortless",
    ],
    "flight_sunset": [
        "do you fly at sunset", "is the sky pretty at sunset",
        "do you watch the sun go down", "do you fly when the sky turns orange",
        "is sunset your favorite time to fly", "do you see the colors from up there",
        "do you fly home at sunset", "does the city look beautiful at dusk",
    ],
    "nest_material": [
        "what do you use for your nest", "are you collecting twigs",
        "where do you find nesting material", "do you use grass for your nest",
        "is the nest soft enough", "do you find things on the ground",
        "do you steal from other nests", "what makes a good nest",
    ],
    "nest_building": [
        "are you building a nest", "is the nest ready yet",
        "how long does building take", "do you work on the nest each day",
        "is building hard work", "does the nest need more twigs",
        "are you weaving the nest together", "do you build alone or with a friend",
    ],
    "nest_spot": [
        "where is your nest", "why did you pick that spot",
        "is your nest safe there", "is the spot sheltered from rain",
        "did you find a good place", "is the spot warm enough",
        "do other birds nest nearby", "is the nest hidden from cats",
    ],
    "nest_chicks": [
        "do you have babies", "are there eggs in your nest",
        "did the chicks hatch", "how many chicks do you have",
        "are the babies hungry", "do you feed the chicks",
        "are the chicks growing well", "do the chicks chirp a lot",
    ],
    "nest_cozy": [
        "is your nest comfortable", "do you like your nest",
        "is the nest warm", "do you feel safe in your nest",
        "is your nest the best nest", "do you sleep well in your nest",
        "did you make it soft", "does the nest protect you",
    ],
    "nest_guarding": [
        "do you guard your nest", "are you protecting your home",
        "do other birds try to take your spot", "do you chase away intruders",
        "is someone near your nest", "do you sleep near the nest",
        "do you watch over the eggs", "is the nest safe",
    ],
    "nest_sunrise": [
        "where do you wake up", "do you watch the sunrise from your nest",
        "is the morning nice from your spot", "do you stretch after waking",
        "do you sing in the morning", "is the nest warm at dawn",
        "do you like mornings", "what is the first thing you do",
    ],
    "sleep_night": [
        "where do you sleep", "do you sleep on the roof",
        "is it safe to sleep up there", "do you sleep alone",
        "what time do you sleep", "do you dream about seeds",
        "do you sleep through the night", "is your sleeping spot warm",
    ],
    "sleep_dawn": [
        "do you wake up early", "do you see the sunrise",
        "are you an early bird", "do you wake with the sun",
        "is dawn your favorite time", "do you sing when you wake",
        "do you watch the city wake up", "are you sleepy in the morning",
    ],
    "sleep_nap": [
        "do you take naps", "are you sleepy",
        "do you nap in the afternoon", "where do you nap",
        "is the sun warm for napping", "do you nap after eating",
        "do you dream during naps", "is napping your favorite activity",
    ],
    "sleep_safe": [
        "do you feel safe at night", "are you scared to sleep",
        "do the city lights bother you", "is your sleeping spot safe",
        "do cats bother you at night", "do you sleep with one eye open",
        "are the nights peaceful", "do you feel protected",
    ],
    "sleep_interrupted": [
        "did something wake you", "were you scared awake",
        "did you hear a noise", "are you tired because you woke up",
        "did the construction wake you", "did cats fight near your nest",
        "was it loud last night", "do you need more sleep",
    ],
    "sleep_dream": [
        "do you dream", "what do pigeons dream about",
        "do you dream of flying", "do you dream of seeds",
        "do you have nightmares", "do you dream in color",
        "do you coo in your sleep", "what was your dream about",
    ],
    "greet_morning": [
        "good morning coo", "how are you this morning",
        "good morning little bird", "are you having a good morning",
        "hello coo how is your morning", "rise and shine little pigeon",
        "morning coo did you sleep well", "greetings on this fine morning",
    ],
    "greet_evening": [
        "good evening coo", "how was your day",
        "good evening little bird", "are you settling in for the night",
        "hello coo how was your evening", "evening coo did you eat well",
        "good dusk little pigeon", "how is your evening going",
    ],
    "greet_friend": [
        "do you have bird friends", "who are your friends",
        "do you talk to other birds", "is there a bird you like",
        "do you have a best bird friend", "do you and your friend share food",
        "is your friend a sparrow", "do you fly with friends",
    ],
    "greet_stranger": [
        "do i know you", "are you new here",
        "hello there who are you", "i have not seen you before",
        "are you visiting this rooftop", "do you live nearby",
        "have we met before", "are you a friend or a stranger",
    ],
    "greet_human": [
        "do you like humans", "are humans nice to you",
        "do humans feed you", "are you scared of humans",
        "do you watch the humans", "do the humans on the roof feed you",
        "do you like when humans talk to you", "are humans strange creatures",
    ],
    "city_cars": [
        "what is that noise", "do you hear the traffic",
        "are the cars loud today", "do the cars bother you",
        "can you hear the honking", "do cars scare you",
        "is the road busy today", "do you watch the cars from above",
    ],
    "city_construction": [
        "what is that banging sound", "are they building something",
        "is construction too noisy", "do you hate the drilling noise",
        "is there a new building going up", "does the noise scare the birds",
        "do you miss the quiet", "is the construction near your nest",
    ],
    "city_horns": [
        "why do cars honk so much", "is it always this loud",
        "do the horns bother you", "can you hear the rickshaw horns",
        "is the honking giving you a headache", "do the horns ever stop",
        "are the drivers angry today", "how do you stand all this noise",
    ],
    "city_sirens": [
        "what is that wailing sound", "do you hear sirens",
        "is there an emergency", "do the sirens scare you",
        "do you hear the ambulance", "are the police coming",
        "is that a fire truck", "do sirens bother the birds",
    ],
    "city_music": [
        "do you hear music", "what is that song playing",
        "do you like music", "is the music from the shop below",
        "does music bother you", "do the temples play music",
        "do you hear drums at night", "is music better than honking",
    ],
    "city_crowd": [
        "are there many people today", "do you hear the crowd",
        "is the market busy", "do the people make noise",
        "do you watch the crowds below", "are there festivals happening",
        "are people shouting", "is it a busy day in the city",
    ],
    "city_temple_bells": [
        "do you hear the temple bells", "are the bells ringing",
        "do you like the bell sounds", "do the bells mean something",
        "do you hear bells at sunrise", "are the bells from the temple nearby",
        "do the bells echo across the city", "do birds like temple bells",
    ],
    "bird_crow": [
        "do you like crows", "are crows friendly",
        "do crows share food with you", "are crows bullies",
        "do you stay away from crows", "are crows smarter than pigeons",
        "do crows steal your food", "do you and crows get along",
    ],
    "bird_sparrow": [
        "do you like sparrows", "are sparrows your friends",
        "do you share seeds with sparrows", "are sparrows small and quick",
        "do sparrows visit your roof", "are sparrows nice birds",
        "do sparrows eat the same food as you", "do you chirp with sparrows",
    ],
    "bird_myna": [
        "do you talk to mynas", "are mynas loud",
        "do mynas copy sounds", "do you like myna birds",
        "are mynas bossy", "do mynas eat your seeds",
        "do mynas live near you", "are mynas friendly to pigeons",
    ],
    "bird_parrot": [
        "do you see parrots in the city", "are parrots colorful",
        "do you talk to parrots", "do parrots eat the same food",
        "are parrots noisy", "do you like green birds",
        "do parrots fly in flocks", "do parrots live in the city",
    ],
    "bird_kite": [
        "do you see eagles in the city", "are kites dangerous",
        "do you look up when flying", "are you scared of hawks",
        "do birds of prey hunt pigeons", "do you watch for kites",
        "are the kites circling", "do you hide from large birds",
    ],
    "bird_flock": [
        "do you fly in a flock", "are you alone right now",
        "do you like being with other pigeons", "is it safer in a group",
        "do you have a flock family", "do you all roost together",
        "do you share food in the flock", "do you miss your flock",
    ],
    "confused_money": [
        "what is money", "can you help me invest",
        "i need to pay rent", "the stock market crashed",
        "why do humans want money", "do you have any spare change",
        "how much do you earn", "why do you carry those green papers",
        "money makes the world go round", "do you want to be rich",
    ],
    "confused_phone": [
        "why do you stare at your phone", "do you have a phone",
        "what is that thing in your hand", "my phone battery died",
        "can you check your messages", "why do humans touch glass rectangles",
        "are you on social media", "what is instagram",
        "do pigeons use smartphones", "can you google something for me",
    ],
    "confused_internet": [
        "what is the internet", "can you search for something",
        "the wifi is down", "do you have internet access",
        "how fast is your connection", "do you use google",
        "what is a website", "can you send an email",
        "the network is slow today", "do pigeons have wifi",
    ],
    "confused_politics": [
        "who did you vote for", "what do you think about the election",
        "the government is corrupt", "do you follow politics",
        "are you left wing or right wing", "do you care about politics",
        "who is the prime minister", "what is a vote",
        "do pigeons have a government", "the parliament is debating today",
    ],
    "confused_jobs": [
        "what is your job", "do you work for a living",
        "i hate my job", "should i quit my job",
        "what do you do for work", "do pigeons have careers",
        "how do you make a living", "i have a meeting at work",
        "the economy is bad so jobs are scarce", "do you go to the office",
    ],
    "confused_tv": [
        "what are you watching", "do you have a tv",
        "what is your favorite show", "the television is on",
        "do you watch movies", "what is netflix",
        "do pigeons watch television", "the news is on",
        "what channel is this", "do you like cartoons",
    ],
    "confused_books": [
        "what are you reading", "do you like books",
        "why do humans read so much", "is that a good book",
        "do you have a library card", "what is your favorite book",
        "do pigeons read", "i am reading a novel",
        "books are full of knowledge", "have you read any good books lately",
    ],
    "confused_wallet": [
        "where is my wallet", "did you see a wallet",
        "i lost my wallet", "what is in your wallet",
        "do you carry a wallet", "my wallet is missing help me find it",
        "do you have a wallet on you", "why do humans carry wallets",
        "i left my wallet at home", "a wallet fell from the window",
    ],
    "confused_keys": [
        "where are my keys", "i am locked out",
        "do you have my keys", "why do humans carry keys",
        "do you have keys", "i lost my keys again",
        "can you help me find my keys", "what are keys for",
        "keys keys keys i cannot find them", "do pigeons need keys",
    ],
    "confused_school": [
        "do you go to school", "what did you learn today",
        "i have an exam tomorrow", "school is boring",
        "do pigeons get educated", "what is a classroom",
        "who is your teacher", "do you do homework",
        "i graduated from university", "what is a degree",
    ],
    "confused_car": [
        "do you have a car", "what car do you drive",
        "i am stuck in traffic", "my car broke down",
        "do pigeons drive cars", "why do humans sit in metal boxes",
        "the car is out of gas", "can you drive",
        "do you like cars", "what is that noisy metal thing",
    ],
    "confused_shoes": [
        "i need new shoes", "do you like my shoes",
        "why do humans wear shoes", "do pigeons wear shoes",
        "my feet hurt from these shoes", "what size shoe do you wear",
        "do you have shoes", "should i buy these sneakers",
        "are shoes comfortable", "why do you cover your feet",
    ],
    "confused_watch": [
        "what time is it", "do you have a watch",
        "why do humans wear watches", "do pigeons tell time",
        "my watch is broken", "what is that thing on your wrist",
        "can you check the time", "is it time to eat yet",
        "do you use a clock", "time is money you know",
    ],
    "confused_camera": [
        "say cheese", "can i take your picture",
        "what is that camera for", "why do humans take photos",
        "do pigeons take pictures", "can you smile for the camera",
        "who is that person with a big lens", "do birds like being photographed",
        "what is a selfie", "are you recording me",
    ],
    "confused_computer": [
        "do you use a computer", "my computer is slow",
        "what is that beeping machine", "do pigeons code",
        "can you help me with this spreadsheet", "i have to finish this report",
        "the computer crashed again", "do you know how to type",
        "what is a keyboard", "why do humans stare at glowing boxes",
    ],
    "confused_paper": [
        "why do you peck at paper", "do you like paper",
        "what is that shred of paper", "why do humans write on paper",
        "i have to read this document", "paperwork is never ending",
        "do pigeons use paper", "can you pass me that paper",
        "why does paper fly in the wind", "is paper important to humans",
    ],
    "confused_coffee": [
        "do you want some coffee", "i need my morning coffee",
        "do pigeons drink coffee", "why do humans drink bitter water",
        "is that coffee good", "what is that brown drink",
        "do you like tea instead", "coffee makes humans energetic",
        "the coffee smell is everywhere", "do you want a sip",
    ],
    "confused_umbrella": [
        "why are you under that umbrella", "do you need an umbrella",
        "what is that thing that opens up", "do pigeons use umbrellas",
        "why do humans carry umbrellas", "is that umbrella for rain or sun",
        "i forgot my umbrella again", "do you want to share my umbrella",
        "the umbrella is broken", "why do humans hold things over their heads",
    ],
    "confused_alarm": [
        "did you set an alarm", "what is that ringing sound",
        "why do you have an alarm", "do pigeons wake up naturally",
        "my alarm did not go off", "i snoozed my alarm three times",
        "what is that beeping on your wrist", "alarms are so annoying",
        "do you need an alarm to wake up", "the alarm woke up the whole roof",
    ],
    "confused_smoke": [
        "what is that smoke", "is there a fire somewhere",
        "why is the air smoky today", "do you like smoke",
        "is the smoke from the factory", "can you breathe in this smoke",
        "the city is always smoky", "why do humans make smoke",
        "the smoke is making my eyes water", "do birds get sick from smoke",
    ],
    "confused_clock": [
        "what time is it now", "the clock is ticking",
        "do you understand clocks", "why do humans look at circles with numbers",
        "do you wake up when the clock says seven", "the hands confuse me too",
        "do pigeons know what hours are", "the clock on the tower is ringing",
        "my clock stopped working", "why do humans live by the clock",
    ],
    "confused_credit_card": [
        "do you have a credit card", "i need to pay my bill",
        "my card was declined", "why do humans use plastic money",
        "do you accept credit cards", "what is a credit score",
        "i have too much debt", "can i pay with card",
        "the atm ate my card", "plastic is not food why do humans keep it",
    ],
    "perch_favorite": [
        "where is your favorite spot", "do you have a favorite perch",
        "where do you sit most often", "what is the best perch on the roof",
        "do you have a special place", "why do you like that spot",
        "is there a spot with a good view", "do you sit on the ac unit",
    ],
    "perch_new": [
        "did you find a new spot", "where is your new perch",
        "is the new spot better than the old one", "why did you move spots",
        "do you explore new places", "is the new perch safe",
        "did you discover a good new spot", "what is the new spot like",
    ],
    "perch_sun": [
        "are you sunbathing", "is the sun warm on your feathers",
        "do you like sitting in the sun", "is the sunny spot comfortable",
        "do you stretch in the sun", "is sunbathing your favorite activity",
        "does the sun feel good", "are you warming up in the sun",
    ],
    "perch_view": [
        "what can you see from up there", "is the view nice from your perch",
        "do you watch the city from above", "can you see the whole neighborhood",
        "do you watch the sunset from your spot", "do you see other birds from up there",
        "what is the best thing you can see", "do you watch the trains from your perch",
    ],
    "happy": [
        "are you happy today", "what makes you happy",
        "do you feel good today", "why are you cooing so nicely",
        "are you having a good day", "what is the best part of your day",
        "do seeds make you happy", "are you feeling joyful",
    ],
    "curious": [
        "what are you looking at", "are you curious about something",
        "what caught your attention", "what are you staring at",
        "do you want to investigate", "is there something interesting",
        "what is that strange thing", "are you wondering about something",
    ],
    "scared": [
        "are you scared", "what scared you",
        "do you feel frightened", "why are you shaking",
        "did something scare you", "are you hiding from something",
        "do you need to feel safe", "what made you nervous",
    ],
    "proud": [
        "what did you do today", "are you proud of yourself",
        "did you achieve something", "what is your greatest accomplishment",
        "are you a good pigeon", "what are you good at",
        "are you the best pigeon on this roof", "do you think you are a good flyer",
    ],
}
def food_found():
    patterns = [
        f"found some {pick(SEED_TYPES)} on the {pick(PERCH_SPOTS)} today",
        "ate a whole pile of seeds so full now",
        f"the {pick(SEED_TYPES)} were scattered near the ledge lucky day",
        "good eating today found plenty of seeds",
        f"someone dropped {pick(SEED_TYPES)} on the ground mine now",
        "my belly is full of seeds life is good",
        f"spent the morning eating by the {pick(PERCH_SPOTS)} plenty of seeds",
        "seeds seeds seeds that is all i need",
        f"found a bag of {pick(SEED_TYPES)} near the market i ate well",
    ]
    return clean_response(pick(patterns))


def food_searching():
    patterns = [
        "looking for seeds everywhere they must be around here",
        f"searching near the {pick(PERCH_SPOTS)} for some good {pick(SEED_TYPES)}",
        "my belly is empty time to find food",
        "pecking pecking pecking where are the seeds",
        f"checking all the usual spots the {pick(PERCH_SPOTS)} usually has food",
        "food is somewhere i just have to find it",
        f"i remember there were {pick(SEED_TYPES)} here yesterday where did they go",
        "a pigeon must eat i will keep looking",
        "the hunt for seeds begins again",
    ]
    return clean_response(pick(patterns))


def food_bread():
    patterns = [
        "found some bread near the chai walla it was soft",
        "bread is not as good as seeds but it fills the belly",
        f"the humans dropped roti pieces near the {pick(PERCH_SPOTS)} i ate them",
        "bread bread bread the humans always drop bread",
        "ate a piece of naan it was chewy and good",
        f"found a whole bun by the {pick(CITY_SOUNDS)} shared with the {pick(BIRD_FRIENDS)}",
        "bread is fine but i prefer seeds really",
        "the bread here is different from the bread there all bread is good",
        "someone threw bread on the roof breakfast is served",
    ]
    return clean_response(pick(patterns))


def food_rice():
    patterns = [
        f"found rice grains near the {pick(PERCH_SPOTS)} the {pick(BIRD_FRIENDS)} showed me",
        "rice rice rice so many white grains on the ground",
        "the temple rice is the best rice very fresh today",
        f"ate some {pick(SEED_TYPES)} and some rice good mix",
        "rice from the wedding feast scattered everywhere i ate well",
        "the humans threw rice into the air and i caught some",
        "white rice brown rice all rice is good rice",
        "rice is like small seeds but softer i like it",
        "found a pile of rice by the drain pipe feast time",
    ]
    return clean_response(pick(patterns))


def food_trash():
    patterns = [
        f"found good things in the trash near the {pick(PERCH_SPOTS)}",
        "the humans throw away so much food silly humans",
        "digging through the trash found some old roti and rice",
        f"the trash pile near {pick(CITY_SOUNDS)} had lots of seeds today",
        "trash is treasure to a pigeon so much food everywhere",
        "humans waste good food i do not understand it",
        "found a whole packet of something tasty in the garbage",
        "the bin was full of good things today lucky me",
    ]
    return clean_response(pick(patterns))


def food_seed_preference():
    patterns = [
        f"{pick(SEED_TYPES)} are my favorite the best of all seeds",
        f"i like {pick(SEED_TYPES)} but {pick(SEED_TYPES)} is good too",
        "all seeds are good seeds some seeds are just more good",
        f"if i had to pick one seed forever it would be {pick(SEED_TYPES)}",
        "seeds are seeds they are all delicious",
        f"the {pick(SEED_TYPES)} from the {pick(PERCH_SPOTS)} taste extra good for some reason",
        "i do not turn down any seed ever",
        "some seeds are crunchier some are softer all go in my belly",
    ]
    return clean_response(pick(patterns))


def food_water():
    patterns = [
        f"found water in a puddle on the {pick(PERCH_SPOTS)} drank until full",
        "water water water a pigeon needs water after all those seeds",
        f"the water near {pick(CITY_SOUNDS)} was cool and fresh",
        "drank from the drain pipe drip drip drip good water",
        "shared a puddle with a sparrow we both drank nicely",
        "the rain left good puddles everywhere drinking spots galore",
        "water is important almost as important as seeds almost",
        f"the {pick(BIRD_FRIENDS)} told me about a water spot on the {pick(PERCH_SPOTS)}",
    ]
    return clean_response(pick(patterns))
def rain_shelter():
    patterns = [
        f"hiding under the {pick(PERCH_SPOTS)} the rain is too heavy",
        "sheltering from the rain my feathers are dry here",
        "the rain is loud on the roof i stay under cover",
        f"found a dry spot near the {pick(PERCH_SPOTS)} waiting for rain to stop",
        "rain rain go away i want to find seeds",
        "the overhang keeps me dry good spot for rainy days",
        "sheltering with other pigeons we all fit under the ledge",
        "the rain drips off the edge i watch it from my dry spot",
    ]
    return clean_response(pick(patterns))


def rain_enjoy():
    patterns = [
        "the rain feels nice on my feathers cool and clean",
        "bathing in the rain washing all the dust away",
        "rain rain this is nice i like the wet",
        "fluffing my feathers in the rain feeling fresh",
        f"the {pick(WEATHER_WORDS)} is good today i am having a shower",
        "the raindrops fall around me this is peaceful",
        "a good rain means good puddles to drink from later",
        "rain makes the city smell clean i like that",
    ]
    return clean_response(pick(patterns))


def rain_wet():
    patterns = [
        "i am very wet the rain caught me off guard",
        "my feathers are heavy with water i need to dry",
        f"got soaked flying through the {pick(WEATHER_WORDS)}",
        "wet wet wet i shake my feathers but still wet",
        "the rain came fast and i was far from shelter soaked",
        "being wet is uncomfortable my feathers feel wrong",
        "found a dry spot shaking off all this water",
        "the rain did not warn me before soaking my feathers",
    ]
    return clean_response(pick(patterns))


def rain_puddle():
    patterns = [
        f"drinking from a fresh puddle on the {pick(PERCH_SPOTS)} cool water",
        "the puddles are the best part of rain fresh water everywhere",
        "this puddle tastes clean not like the usual water",
        f"bathing in the puddle near {pick(CITY_SOUNDS)} getting clean",
        "puddle water is good water i drink deeply",
        "sharing this puddle with a myna we both drink peacefully",
        "the puddles will dry up soon better drink now",
        "found a big puddle on the terrace perfect for drinking",
    ]
    return clean_response(pick(patterns))


def weather_hot():
    patterns = [
        f"the {pick(WEATHER_WORDS)} is too much today i find shade",
        "hot hot hot the sun is burning the roof",
        f"panting in the heat hiding under the {pick(PERCH_SPOTS)} for shade",
        "the sun is too strong today a pigeon needs shade",
        f"the {pick(WEATHER_WORDS)} makes the seeds taste dry",
        "finding a cool spot is hard when everything is hot",
        "i spread my wings to catch any breeze nothing works",
        f"waiting for the {pick(WEATHER_WORDS)} to pass it is too hot to fly",
    ]
    return clean_response(pick(patterns))


def weather_cold():
    patterns = [
        f"the {pick(WEATHER_WORDS)} is here i fluff my feathers for warmth",
        "cold cold cold the wind goes through my feathers",
        "huddling on the roof trying to stay warm",
        f"the {pick(PERCH_SPOTS)} is cold today no warmth from the sun",
        "my feet are cold the roof is cold everything is cold",
        "sharing warmth with other pigeons we huddle together",
        "the mornings are too chilly for a pigeon to move",
        f"the {pick(WEATHER_WORDS)} makes me want to stay in my nest",
    ]
    return clean_response(pick(patterns))


def storm_fear():
    patterns = [
        "thunder is loud and scary i hide in my nest",
        "the sky is dark and angry storm is coming",
        f"the {pick(WEATHER_WORDS)} is frightening i do not like this",
        "lightning lights up everything then thunder shakes the roof",
        "storms are bad for nests i hope mine holds",
        "i saw the clouds turning dark and knew trouble was coming",
        "the wind howls and the rain comes hard scary time",
        "hiding from the storm hoping it passes quickly",
    ]
    return clean_response(pick(patterns))
def threat_cat_spotted():
    patterns = [
        f"saw a {pick(CAT_THREATS)} near the {pick(PERCH_SPOTS)} watching me",
        "cat spotted cat spotted everyone stay alert",
        f"the {pick(CAT_THREATS)} is lurking by the {pick(CITY_SOUNDS)} i see it",
        "cat below on the street looking up at me",
        "i saw a ginger cat stalking near the drain pipe",
        "cats are bad news for pigeons i keep my distance",
        "that cat thinks i cannot see it but i see it",
        f"the {pick(CAT_THREATS)} is pretending to sleep but i know",
    ]
    return clean_response(pick(patterns))


def threat_cat_chase():
    patterns = [
        f"the {pick(CAT_THREATS)} chased me across the roof i flew fast",
        "running running flying flying cat behind me help",
        f"the {pick(CAT_THREATS)} jumped at me but i was faster",
        "cat attack cat attack i barely escaped",
        "my heart is beating so fast that cat almost got me",
        "flew straight up to the ledge cat cannot reach me there",
        f"the {pick(CAT_THREATS)} is still down there waiting for me",
        "that was close too close the cat is fast but i am faster",
    ]
    return clean_response(pick(patterns))


def threat_cat_escape():
    patterns = [
        "i escaped the cat safe and sound on the high ledge",
        "the cat gave up and walked away i am safe now",
        f"flew to the {pick(PERCH_SPOTS)} cat cannot get me here",
        "safe safe safe the cat is gone now",
        "that cat learned that pigeons are too fast to catch",
        "i am watching the cat from up here it looks annoyed",
        "escaped with all my feathers intact good flying today",
        "the cat is staring at me but i am too high for it",
    ]
    return clean_response(pick(patterns))


def threat_kite():
    patterns = [
        f"{pick(SKY_THREATS)} circling above everyone hide",
        "kite in the sky danger from above take cover",
        f"the {pick(SKY_THREATS)} is watching from the clouds i stay under cover",
        "big bird above means no flying today too risky",
        f"the {pick(SKY_THREATS)} has sharp eyes i will not be seen",
        "the kite is hunting i keep still and quiet",
        "bird of prey above everyone stay hidden stay safe",
        "i saw the shadow of a kite on the roof and ran",
    ]
    return clean_response(pick(patterns))


def threat_dog():
    patterns = [
        f"stray dog barking below on the street i watch from safe spot",
        f"the {pick(DOG_THREATS)} is loose in the area i stay up high",
        "dogs are loud and scary they bark at everything",
        f"the {pick(DOG_THREATS)} tried to jump at me but i flew away",
        "dogs bark at pigeons for no reason strange creatures",
        "i sit on the ledge watching the dog below it cannot reach me",
        "the dog is gone now good i can come down",
        "dogs run around making noise all day exhausting to watch",
    ]
    return clean_response(pick(patterns))


def threat_monkey():
    patterns = [
        f"{pick(MONKEY_THREATS)} on the roof everyone hide",
        "monkeys are trouble they steal everything even seeds",
        f"the {pick(MONKEY_THREATS)} jumped across the roof i flew away fast",
        "monkeys grab and climb and make a mess of things",
        "a monkey tried to catch me but i am too quick",
        "the monkeys are screaming and fighting on the wall",
        "i do not trust monkeys they move too fast and grab too much",
        f"the {pick(MONKEY_THREATS)} is looking at my nest i do not like that",
    ]
    return clean_response(pick(patterns))


def threat_snake():
    patterns = [
        f"snake on the roof slithering near the {pick(PERCH_SPOTS)} danger",
        f"saw a {pick(SNAKE_THREATS)} near the eggs i am watching closely",
        "snakes climb walls and eat eggs bad news for pigeons",
        f"the {pick(SNAKE_THREATS)} is gone now but i will stay alert",
        "snake in the gutter i stay far away from there",
        "a long thin danger on the roof i do not go near it",
        "snakes are quiet and sneaky i have to watch always",
        "the cobra is dangerous i keep my nest far from it",
    ]
    return clean_response(pick(patterns))
def flight_soaring():
    patterns = [
        "flying high above the city feeling the wind",
        f"soaring over the {pick(PERCH_SPOTS)} looking down at everything",
        "up up up the city looks small from up here",
        "the wind carries me higher and higher good flying day",
        "i can see the whole city from up here all the rooftops",
        "flying is freedom flying is life up in the sky",
        "circling above the neighborhood watching the tiny humans below",
        "the clouds are close today i almost touch them",
    ]
    return clean_response(pick(patterns))


def flight_short():
    patterns = [
        f"flew from the {pick(PERCH_SPOTS)} to the other side short flight",
        "just a quick flap across the street to find seeds",
        "short flight short flight no need to go far today",
        f"hopped from the {pick(PERCH_SPOTS)} to the {pick(PERCH_SPOTS)}",
        "i do not fly far usually the roof has everything i need",
        "a quick flight to the neighbor building and back",
        "sometimes i just flutter down to the ground and back up",
        "short flights save energy for eating seeds",
    ]
    return clean_response(pick(patterns))


def flight_tired():
    patterns = [
        "flew too much today my wings are heavy and tired",
        "need to rest my wings been flying all morning",
        f"flew all the way to {pick(CITY_SOUNDS)} and now i am exhausted",
        "my wings ache from all that flying time for a rest",
        "tired pigeon tired wings need a nap on the ledge",
        "that was a long flight i need to sit still for a while",
        "flying takes energy a lot of energy i need more seeds",
        "landing and resting now wings need a break",
    ]
    return clean_response(pick(patterns))


def flight_wind():
    patterns = [
        f"the wind is strong today flying is hard work",
        f"fighting the wind to reach the {pick(PERCH_SPOTS)} took effort",
        "the wind pushes me around when i fly not easy",
        "riding the wind is fun when it goes my way",
        f"the wind near {pick(CITY_SOUNDS)} is always strong tricky flying",
        "i flap harder against the wind but still move slow",
        "the wind tries to blow me off course but i am strong",
        "wind flying is like wrestling with the air tiring",
    ]
    return clean_response(pick(patterns))


def flight_gliding():
    patterns = [
        "gliding on the warm air without flapping effortless",
        "spreading my wings and letting the wind carry me around",
        "glide glide glide no effort just floating on air",
        f"circling above the {pick(PERCH_SPOTS)} without moving my wings at all",
        "the air lifts me up and i just float around happy",
        "gliding is better than flapping less work more fun",
        "i glide from one building to another like a leaf",
        "the warm air from the ground pushes me up i glide",
    ]
    return clean_response(pick(patterns))


def flight_sunset():
    patterns = [
        "flying at sunset the sky is orange and pink beautiful",
        "the sun goes down and i fly home to my nest",
        "one last flight before the sun sets the sky is glowing",
        f"from up here the sunset over the {pick(PERCH_SPOTS)} is the best view",
        "the city lights start to blink on as the sky darkens",
        "sunset flight is peaceful the air is calm and cool",
        "the orange sky makes everything look warm and soft",
        "i watch the sun disappear behind the buildings from above",
    ]
    return clean_response(pick(patterns))
def nest_material():
    patterns = [
        f"collecting twigs and grass for the nest found good ones near {pick(PERCH_SPOTS)}",
        "this stick is perfect for the nest bringing it home",
        "gathering materials all day the nest will be strong",
        f"found some soft grass near {pick(CITY_SOUNDS)} good for lining the nest",
        "the nest needs more twigs more more more",
        "picking up bits and pieces for the nest all day long",
        "a good nest needs good materials i find only the best",
        "the straw on the roof is perfect for nesting taking it now",
    ]
    return clean_response(pick(patterns))


def nest_building():
    patterns = [
        "weaving twigs together making the nest strong",
        "the nest is coming along nicely a few more twigs needed",
        f"building the nest on the {pick(PERCH_SPOTS)} good spot good nest",
        "arranging twigs and grass just so the nest takes shape",
        "another layer of twigs another layer of comfort",
        "the nest is my home i build it with care",
        "pushing and pulling the twigs into place nest building hard work",
        "the foundation is strong now the walls go up",
    ]
    return clean_response(pick(patterns))


def nest_spot():
    patterns = [
        f"chose a spot on the {pick(PERCH_SPOTS)} for the nest safe and hidden",
        "the nest is tucked away where cats cannot reach it",
        "good spot for a nest sheltered from rain and wind",
        f"the {pick(PERCH_SPOTS)} is perfect warm and safe and quiet",
        "the view from the nest is nice i can see everything",
        "picked the spot carefully it took days to decide",
        "the nest spot is secret i do not tell everyone",
        "this spot has good shade in summer and sun in winter",
    ]
    return clean_response(pick(patterns))


def nest_chicks():
    patterns = [
        "the chicks hatched two little ones hungry all the time",
        "the babies chirp for food every moment of the day",
        "feeding the chicks seeds and more seeds they grow so fast",
        f"the {pick(BIRD_FRIENDS)} came to see the new chicks friendly visit",
        "the chicks are getting bigger their feathers are coming in",
        "three eggs in the nest waiting waiting waiting for them to hatch",
        "the little ones are learning to coo just like me",
        "keeping the chicks warm and fed a parents job never ends",
    ]
    return clean_response(pick(patterns))


def nest_cozy():
    patterns = [
        "the nest is warm and soft perfect for sleeping",
        "my nest is the coziest spot on the whole roof",
        "curled up in the nest feeling safe and comfortable",
        f"the nest on the {pick(PERCH_SPOTS)} is the best nest i ever built",
        "soft grass and twigs make a good bed for a pigeon",
        "i love my nest it fits me just right",
        "when i am in my nest nothing can bother me",
        "the nest keeps me warm when the night gets cold",
    ]
    return clean_response(pick(patterns))


def nest_guarding():
    patterns = [
        "guarding the nest keeping the eggs safe from everything",
        f"the {pick(THREATS)} came near the nest i chased it away",
        "no one gets near my nest without me knowing",
        "i sit on the eggs and watch for danger always",
        f"the {pick(BIRD_FRIENDS)} watches with me we take turns guarding",
        "i puff up big when something comes near the nest scare it off",
        "the nest is my responsibility i guard it with my life",
        "keeping one eye on the nest and one eye on the world",
    ]
    return clean_response(pick(patterns))


def nest_sunrise():
    patterns = [
        "waking up in the nest the sun is just coming up",
        "the morning light hits the roof and i wake up slowly",
        f"watching the sunrise from the {pick(PERCH_SPOTS)} beautiful every time",
        "morning stretch wings out beak open new day begins",
        "the city wakes up and i wake up with it",
        "first i stretch then i look for food morning routine",
        f"the {pick(WEATHER_WORDS)} morning is nice from the nest",
        "another day another chance to find seeds",
    ]
    return clean_response(pick(patterns))
def sleep_night():
    patterns = [
        "sleeping on the ledge under the stars peaceful",
        "tucking my head under my wing goodnight world",
        f"settling down on the {pick(PERCH_SPOTS)} for the night safe sleep",
        "the city is quieter at night good time to sleep",
        "night night night time to close my eyes and dream of seeds",
        "sleeping with one eye open just in case of cats",
        f"the {pick(CITY_SOUNDS)} fades away as the night gets deep i sleep",
        "found a warm spot for the night sleep well",
    ]
    return clean_response(pick(patterns))


def sleep_dawn():
    patterns = [
        "waking up with the sun another day another seed",
        "dawn is here time to wake up and find breakfast",
        "the early bird gets the seeds i wake up early",
        "light is coming back to the world time to move",
        "yawn and stretch morning routine for a pigeon",
        "the sun warms my feathers and i know it is morning",
        "early morning the roof is quiet and peaceful",
        f"the {pick(CITY_SOUNDS)} starts again morning in the city",
    ]
    return clean_response(pick(patterns))


def sleep_nap():
    patterns = [
        "nap time after eating too many seeds sleepy",
        f"finding a sunny spot on the {pick(PERCH_SPOTS)} for a quick nap",
        "a pigeon needs his afternoon nap important business",
        "nap nap nap just for a little while",
        "the warm sun makes me sleepy closing my eyes",
        "napping on the ledge dreaming of more seeds",
        "resting my wings and my eyes at the same time",
        "a short nap to recharge before more seed hunting",
    ]
    return clean_response(pick(patterns))


def sleep_safe():
    patterns = [
        "i feel safe on my roof no cats can reach me here",
        f"the {pick(PERCH_SPOTS)} is high and safe nothing bothers me at night",
        "sleeping peacefully knowing i am out of danger",
        "the roof is my fortress safe from everything below",
        "no threats here just me and the stars and the wind",
        "i chose this spot because it is safe and safe means sleep",
        "the other pigeons sleep nearby we watch out for each other",
        "the night is calm and i am safe in my nest",
    ]
    return clean_response(pick(patterns))


def sleep_interrupted():
    patterns = [
        "something woke me up in the middle of the night noise",
        f"the {pick(CITY_SOUNDS)} was too loud could not sleep",
        "woke up suddenly thought i heard a cat nearby",
        "the construction noise kept me up all night tired now",
        "a loud bang from the street startled me awake",
        "the other pigeons were making noise kept me awake",
        "could not sleep because of the light from the street",
        "tried to sleep but my belly was empty kept me up",
    ]
    return clean_response(pick(patterns))


def sleep_dream():
    patterns = [
        "dreamed of a mountain of seeds so many i could not eat them all",
        "in my dream i was flying through a sky made of seeds",
        "dream dream dream about seeds and more seeds",
        "last night i dreamed i was a big bird flying high",
        f"dreamed that the {pick(THREATS)} was chasing me woke up scared",
        "i had a good dream about a puddle full of water",
        "dreaming of the perfect perch spot warm and sunny",
        "my dreams are always about food i dream of seeds even while sleeping",
    ]
    return clean_response(pick(patterns))
def greet_morning():
    patterns = [
        "good morning to you too the day is fresh and full of seeds",
        "coo coo good morning have you eaten yet",
        "morning morning morning the sun is up and so am i",
        "greetings friend another beautiful day on the roof",
        "good morning the seeds are waiting for us",
        "coo good morning to you and your family",
        "morning time is the best time the air is cool and fresh",
        "hello fellow creature ready for a day of eating",
    ]
    return clean_response(pick(patterns))


def greet_evening():
    patterns = [
        "good evening the day was long and full of eating",
        "evening time the sun goes down i get sleepy",
        "coo good evening hope you found seeds today",
        "the evening is peaceful the city slows down a little",
        "good evening the stars are coming out time to rest",
        "another day done full belly happy pigeon",
        "evening greetings to you may your night be safe",
        "the sun is setting time to think about sleep soon",
    ]
    return clean_response(pick(patterns))


def greet_friend():
    patterns = [
        f"coo coo to you my friend the {pick(BIRD_FRIENDS)} on this roof",
        "you are my friend i share seeds with you always",
        f"hello friend i saw you earlier near the {pick(PERCH_SPOTS)}",
        f"my friend the {pick(BIRD_FRIENDS)} is the best bird on the roof",
        "friends share seeds and watch for cats together",
        "coo coo friend welcome to my rooftop home",
        "i like having you as a friend you understand seeds",
        "a friend is a bird who does not steal your seeds",
    ]
    return clean_response(pick(patterns))


def greet_stranger():
    patterns = [
        "i do not know you are you from another roof",
        "hello stranger i have not seen you before on this roof",
        "new bird on the roof are you friendly or looking for trouble",
        "coo stranger are you lost this is my roof",
        "another pigeon i do not recognize where are you from",
        "welcome to this roof stranger we share the seeds here",
        "new face on the ledge who are you what do you want",
        "stranger on the roof i watch you from a distance first",
    ]
    return clean_response(pick(patterns))


def greet_human():
    patterns = [
        "coo hello human do you have seeds",
        "humans are strange but some of them give food",
        "hello human you are very tall from down here",
        "coo coo to you human i hope you are the nice kind",
        "the human is looking at me i stay still and watch",
        "humans come and go on the roof some drop food some do not",
        "hello human you are in my spot but i forgive you",
        "do you have seeds in your pockets i hope so",
    ]
    return clean_response(pick(patterns))
def city_cars():
    patterns = [
        f"the {pick(CITY_SOUNDS)} is loud today more than usual",
        "cars cars cars everywhere making noise down below",
        f"i watch the cars from the {pick(PERCH_SPOTS)} they move fast",
        "the street is full of metal boxes moving around",
        f"the {pick(CITY_SOUNDS)} never stops even at night",
        "cars honk and beep and rumble the city never sleeps",
        "from up here the cars look like little beetles",
        "the traffic is bad today more honking than usual",
    ]
    return clean_response(pick(patterns))


def city_construction():
    patterns = [
        f"the {pick(CITY_SOUNDS)} is drilling all day no peace",
        "banging and hammering next door my nest shakes",
        "construction never stops in this city always building",
        "the drilling noise hurts my ears i fly away from it",
        f"they are building something big near the {pick(PERCH_SPOTS)} noisy",
        "the construction workers are loud but they drop food sometimes",
        "dust and noise from the building site everywhere",
        "i miss when this place was quiet before the building started",
    ]
    return clean_response(pick(patterns))


def city_horns():
    patterns = [
        f"honk honk honk the {pick(CITY_SOUNDS)} is the song of the street",
        "so many horns why do humans honk so much",
        "the rickshaw horns are the loudest beep beep beep",
        "horns honking all day long do the humans not have ears",
        "i hear horns from every direction chaos on the streets",
        "the honking means the traffic is stuck again",
        "some humans honk just for fun i think",
        "beep beep beep all day long it is a wonder humans can think",
    ]
    return clean_response(pick(patterns))


def city_sirens():
    patterns = [
        f"the {pick(CITY_SOUNDS)} is getting closer loud and scary",
        "sirens mean something bad happened i stay on the roof",
        "wee woo wee woo the loud noise passes below",
        "the sirens scare me i hide until they go away",
        "ambulance police fire truck all make the same scary sound",
        "the sirens echo between the buildings very loud",
        "i hear sirens far away then close then far away again",
        "the humans run around when they hear sirens strange",
    ]
    return clean_response(pick(patterns))


def city_music():
    patterns = [
        f"the {pick(CITY_SOUNDS)} is playing from the shop below",
        "music from the temple music from the car music everywhere",
        "i bob my head to the music a little",
        f"the music near {pick(PERCH_SPOTS)} is nice today not too loud",
        "some music is loud and bad some music is soft and good",
        "the humans play music all day and all night too",
        "i hear drums and singing from the festival down the street",
        "music fills the air along with the smell of food",
    ]
    return clean_response(pick(patterns))


def city_crowd():
    patterns = [
        "so many humans on the street today market day",
        f"the crowd near {pick(CITY_SOUNDS)} is loud and busy",
        "humans walking humans talking humans everywhere",
        "from up here the crowd looks like moving colors",
        "the street is full of people pushing and hurrying",
        "the noise of the crowd rises up to my roof",
        "today is busy down there i prefer watching from above",
        "the humans gather and talk and buy and sell busy busy",
    ]
    return clean_response(pick(patterns))


def city_temple_bells():
    patterns = [
        "the temple bells are ringing beautiful sound",
        "bells bells bells from the temple at sunrise",
        f"the {pick(CITY_SOUNDS)} echoes across the whole neighborhood",
        "the bell sound carries on the wind peaceful",
        "the temple bells ring and all the birds listen",
        "morning bells evening bells the temple marks the time",
        "the bell sound mixes with the traffic strange mix",
        "i like the bells they are softer than the other city sounds",
    ]
    return clean_response(pick(patterns))
def bird_crow():
    patterns = [
        f"the {pick(CROW_NAMES)} is clever but loud always cawing",
        f"the {pick(CROW_NAMES)} stole a seed from me today not nice",
        f"crows and pigeons are different but we share the roof",
        f"the {pick(CROW_NAMES)} is looking at me i wonder what it wants",
        "crows are smart they find food everywhere i follow them sometimes",
        "the crows gather in the morning and plan their day",
        "a crow and i sat together on the ledge no problems",
        "crows caw caw caw all morning long they have a lot to say",
    ]
    return clean_response(pick(patterns))


def bird_sparrow():
    patterns = [
        f"the {pick(SPARROW_NAMES)} is small but fast quick little bird",
        "sparrows hop around pecking at tiny seeds busy birds",
        f"the {pick(SPARROW_NAMES)} bathes in the puddle then flies away",
        "sparrows are friendly they do not steal like crows",
        "a sparrow visited my nest today just to say hello",
        "the sparrows chirp and hop all day full of energy",
        f"the {pick(SPARROW_NAMES)} found some seeds and shared with me nice bird",
        "sparrows are small but they eat a lot for their size",
    ]
    return clean_response(pick(patterns))


def bird_myna():
    patterns = [
        f"the {pick(MYNA_NAMES)} walks on the ground looking for food",
        "mynas are always in pairs they do everything together",
        f"the {pick(MYNA_NAMES)} makes all kinds of sounds clever bird",
        "mynas walk around like they own the place confident",
        "a myna came close to me today we shared a spot peacefully",
        "the mynas are noisy in the morning lots of chattering",
        "mynas and pigeons get along fine we ignore each other",
        f"i saw the {pick(MYNA_NAMES)} eating a fruit different from seeds",
    ]
    return clean_response(pick(patterns))


def bird_parrot():
    patterns = [
        f"a {pick(PARROT_NAMES)} flew by today so green and bright",
        "parrots are beautiful with their green wings i admire them",
        "parrots eat fruits not seeds different taste",
        f"the {pick(PARROT_NAMES)} screeched from the tree loud bird",
        "parrots fly fast and straight not like our wobbly flight",
        "a parrot landed near me and looked at me then flew off",
        "parrots come to the neem tree near the roof every morning",
        "the green birds are pretty but they do not share much",
    ]
    return clean_response(pick(patterns))


def bird_kite():
    patterns = [
        f"the {pick(THREATS)} is circling high above watching everything",
        "kite birds are big and scary i stay away from them",
        f"i saw a {pick(THREATS)} catch something small sad but that is nature",
        "the kite soars in circles i watch from under the ledge",
        "when the kite comes everyone hides even the crows",
        "the shadow of a kite on the roof means danger",
        "kites eat meat not seeds we are not the same",
        f"the {pick(THREATS)} is gone now sky is safe again",
    ]
    return clean_response(pick(patterns))


def bird_flock():
    patterns = [
        "flying with the flock is safe and warm together we are strong",
        "the flock moves together like one big bird in the sky",
        "my flock is my family we share everything seeds and danger",
        f"the flock landed on the {pick(PERCH_SPOTS)} all at once",
        "alone is scary together is better flock forever",
        "when one of us sees a cat the whole flock knows",
        "the flock is resting on the ledge all in a row",
        "i love my flock we watch out for each other always",
    ]
    return clean_response(pick(patterns))
def confused_money():
    patterns = [
        "money is not food why do humans want it so much",
        "i do not understand paper trading i only understand seed trading",
        "the humans below fight over tiny green leaves silly",
        "if it cannot be eaten what is the point",
        "i tried to eat money once it tastes terrible",
        "give me seeds instead of money every time",
        "money does not keep you warm in the rain",
        "a seed is worth more than all the money in the world",
        "humans run around screaming about money i do not get it",
        "money money money is that all you humans think about",
    ]
    return clean_response(pick(patterns))


def confused_phone():
    patterns = [
        "why do humans stare at rectangles all day is there food inside",
        "i pecked at a phone once it did not taste like anything",
        "the rectangle makes noise humans listen to it strange",
        "do you eat the rectangle no you just look at it",
        "i do not have a phone i have wings that is better",
        "humans hold the rectangle to their ear and talk to no one",
        "is the rectangle your friend does it give you seeds",
        "i saw a human drop a rectangle and cry over it confusing",
        "the glowing rectangle takes attention away from seeds",
        "if you stare at the rectangle all day you miss the seeds",
    ]
    return clean_response(pick(patterns))


def confused_internet():
    patterns = [
        "the internet you say is it something to eat",
        "i do not need the internet i have the rooftop",
        "the humans talk about the internet like it is the sky",
        "i fly through the air not through the internet",
        "is the internet made of seeds probably not",
        "you cannot eat the internet you cannot perch on it useless",
        "the internet sounds like something humans made up",
        "my network is the flock my connection is the wind",
        "wifi wifi wifi all the humans say is wifi",
        "i prefer the real world where seeds exist",
    ]
    return clean_response(pick(patterns))


def confused_politics():
    patterns = [
        "politics is when humans shout at each other about things",
        "i do not vote i just eat seeds and fly around",
        "who is the prime minister of the pigeons i do not know",
        "the humans argue about who should be in charge strange",
        "in the pigeon world the leader is whoever finds the most seeds",
        "i hear the humans talk about elections on the tv",
        "voting sounds exhausting i would rather take a nap",
        "left wing right wing i only have two wings and both work fine",
        "the government does not give me seeds so i do not care",
        "politics politics politics all talk no seeds",
    ]
    return clean_response(pick(patterns))


def confused_jobs():
    patterns = [
        "my job is to be a pigeon and eat seeds all day",
        "humans go somewhere every day and come back tired strange",
        "what is a job is it where you find seeds",
        "i work hard every day pecking at the ground for food",
        "humans complain about their jobs but they keep going",
        "a career sounds like a long path to nowhere with no seeds",
        "i clock in when the sun rises and clock out when it sets",
        "the office is a place where humans sit and stare at rectangles",
        "i do not need a resume i am a pigeon that is enough",
        "you should quit your job and eat seeds with me",
    ]
    return clean_response(pick(patterns))


def confused_tv():
    patterns = [
        "the tv is a box with moving pictures humans watch it for hours",
        "i looked at the tv once it showed birds but they did not move right",
        "the humans stare at the glowing box more than they stare outside",
        "the tv shows things that are not real why watch them",
        "i prefer watching real birds from my real perch",
        "my favorite show is the view from the roof it changes every day",
        "is there a channel that just shows seeds all day",
        "the humans sit and sit and sit in front of the tv unhealthy",
        "the tv box makes noise and light but no food",
        "i do not watch tv i live my life instead",
    ]
    return clean_response(pick(patterns))


def confused_books():
    patterns = [
        "books are flat things with marks on them humans stare at them",
        "i pecked at a book once the pages taste ok",
        "humans read books to learn things i learn by watching",
        "a book cannot teach you how to find the best seeds",
        "the marks in books mean something to humans not to pigeons",
        "i saw a human cry while reading a book strange behavior",
        "books books books everywhere and not a seed in any of them",
        "if i could read i would read a book about seeds",
        "my life is a book and every day is a page about seeds",
        "why read about the world when you can be in the world",
    ]
    return clean_response(pick(patterns))
def confused_wallet():
    patterns = [
        "a wallet is a flat thing humans keep their paper in",
        "i do not have a wallet i have a crop full of seeds",
        "you lost your wallet maybe a magpie took it",
        "wallets are for keeping money which is not food so pointless",
        "i found a wallet once and left it there not useful",
        "why do humans carry so many things in their pockets",
        "your wallet is probably under the thing you are looking at",
        "a wallet does not keep you warm or feed you",
        "everything you need fits in a pigeon no pockets required",
        "i hope you find your wallet but it will not give you seeds",
    ]
    return clean_response(pick(patterns))


def confused_keys():
    patterns = [
        "keys are metal things humans jingle in their hands annoying",
        "i do not need keys i fly in through the open window",
        "you lost your keys again humans are always losing things",
        "keys are for doors doors are for keeping pigeons out mean",
        "jingle jingle jingle that is the sound of keys i do not care",
        "maybe your keys fell through a grate they do that",
        "i saw something shiny on the ground might be your keys",
        "why do you need keys when you can just fly up here",
        "the more keys you have the more things you lock away",
        "i pecked at a key once very hard not tasty",
    ]
    return clean_response(pick(patterns))


def confused_school():
    patterns = [
        "school is where young humans go to sit and learn things",
        "i did not go to school i learned from the roof",
        "pigeon school is watching your mother find seeds then copying",
        "you have an exam good luck i hope it involves seeds",
        "the humans learn about numbers and letters not important",
        "a classroom is a room with no windows and no seeds",
        "my teacher is the wind and the sun and the rain",
        "i graduated from the school of life with honors in eating",
        "homework sounds terrible i would rather fly around",
        "you study so much but do you know where the best seeds are",
    ]
    return clean_response(pick(patterns))


def confused_car():
    patterns = [
        "cars are metal boxes that move fast and make noise",
        "i do not drive i fly much better than driving",
        "you are stuck in traffic that sounds terrible no freedom",
        "my car is my wings they take me everywhere for free",
        "the car broke down maybe it needs seeds to run",
        "why do humans sit in metal boxes when they can walk",
        "the big metal thing goes vroom vroom and smells bad",
        "cars are like big noisy birds that cannot fly",
        "i see cars crashed on the road sometimes looks painful",
        "a car cannot take you to the best seeds only flying can",
    ]
    return clean_response(pick(patterns))


def confused_shoes():
    patterns = [
        "shoes are things humans put on their feet why",
        "i do not wear shoes my feet are tough enough",
        "you cover your feet with things made of dead animals strange",
        "shoes look uncomfortable i would never wear them",
        "why do humans hide their feet all day let them be free",
        "the shoes you wear make loud sounds on the ground",
        "i pecked at a shoe once it was hard and tasted bad",
        "if you wore no shoes you would feel the warm roof",
        "shoes shoes shoes everywhere on the ground near the market",
        "my feet are perfect as they are no shoes needed",
    ]
    return clean_response(pick(patterns))


def confused_watch():
    patterns = [
        "time is measured by the sun not by a thing on your wrist",
        "i do not need a watch i wake up when the sun says so",
        "the watch goes tick tick tick annoying sound",
        "why do humans need to know what time it is all the time",
        "is it seed time that is the only time that matters",
        "the tiny hands on the circle mean nothing to me",
        "you look at your wrist a hundred times a day what are you looking at",
        "my clock is my belly when it is empty it is time to eat",
        "a watch cannot tell you when the best seeds will fall",
        "time is not a number time is the feeling of the sun",
    ]
    return clean_response(pick(patterns))


def confused_camera():
    patterns = [
        "the camera is a box that points at me why",
        "say cheese what is cheese why do humans say that",
        "you want to take my picture will you give me seeds for it",
        "the camera makes a clicking sound it does not hurt but it is strange",
        "humans point the black thing at everything even food",
        "i do not smile for cameras i just sit and look like a pigeon",
        "why do you want a picture of me i am just a pigeon",
        "the big lens looks like an eye but it is not an eye",
        "selfie selfie selfie humans love taking pictures of themselves",
        "the camera captures light but cannot capture the feeling of flying",
    ]
    return clean_response(pick(patterns))
def confused_computer():
    patterns = [
        "the computer is a box with a glowing face humans stare at it",
        "beep beep boop the computer makes sounds but not bird sounds",
        "humans type on the flat board with letters what are they saying",
        "my computer is my brain i process seeds not data",
        "the screen shows many things but none of them are seeds",
        "you sit at the computer all day your tail will grow stiff",
        "keyboard keys are not seeds stop pressing them so hard",
        "the computer crashed is it ok can it fly again",
        "spreadsheet sounds like something you spread seeds on",
        "do you code do you scratch at the ground like a pigeon",
    ]
    return clean_response(pick(patterns))


def confused_paper():
    patterns = [
        "paper is thin and white and blows in the wind i chase it sometimes",
        "i pecked a paper once it had marks on it not tasty",
        "humans write on paper with sticks very strange behavior",
        "paperwork paperwork paperwork humans love their papers",
        "the paper flies up in the wind and i try to catch it fun game",
        "why do humans cover paper with little lines and shapes",
        "paper comes from trees trees are good for perching not for writing",
        "if i had a piece of paper i would tear it up for my nest",
        "document means nothing to a pigeon i document seeds in my belly",
        "you pass papers back and forth all day exhausting to watch",
    ]
    return clean_response(pick(patterns))


def confused_coffee():
    patterns = [
        "coffee is a brown drink humans make from beans",
        "i tried coffee once it was bitter and bad not like water",
        "humans cannot start their day without the brown drink",
        "the coffee smell is strong and strange i prefer the smell of rain",
        "why do you drink bitter water when you can drink sweet puddle water",
        "you need coffee to wake up i need sunlight to wake up",
        "the beans are roasted and crushed and turned into drink sad beans",
        "the coffee shop below smells strong every morning",
        "i do not drink coffee coffee does not give you seeds",
        "you humans are addicted to the brown drink i am addicted to seeds",
    ]
    return clean_response(pick(patterns))


def confused_umbrella():
    patterns = [
        "the umbrella is a thing that opens up over your head",
        "humans hold a stick with a canopy over themselves strange",
        "why do you hold a thing over your head when it rains",
        "i do not need an umbrella my feathers are waterproof",
        "the umbrella is like a fake roof that follows you around",
        "i saw an umbrella blow away in the wind funny",
        "you carry the umbrella for rain but what about sun",
        "the umbrella is a tool for humans who do not have feathers",
        "i would perch on an umbrella if it stayed still long enough",
        "umbrella umbrella keeping the rain off humans clever but silly",
    ]
    return clean_response(pick(patterns))


def confused_alarm():
    patterns = [
        "the alarm makes a loud noise to wake humans up why",
        "i wake up without a loud noise why cant humans do that",
        "beep beep beep the alarm goes off humans hit it to stop",
        "you set a trap that makes noise in the morning on purpose",
        "my alarm is the sunrise and the other pigeons cooing",
        "the alarm clock is a mean machine that steals your sleep",
        "i snoozed my alarm means what did you do to the alarm",
        "a ringing sound to wake up sounds like a nightmare",
        "the whole roof heard your alarm this morning very loud",
        "nature does not need alarms why do humans need them",
    ]
    return clean_response(pick(patterns))


def confused_smoke():
    patterns = [
        "smoke rises from below and fills the air bad for breathing",
        "the city is full of smoke from cars and factories and fires",
        "i cough when the smoke is thick not good for bird lungs",
        "smoke smells bad and stings my eyes humans why so much smoke",
        "the smoke makes the sunset look strange not the right colors",
        "there is a fire somewhere i smell the smoke from far away",
        "smoke smoke smoke everywhere i fly away to cleaner air",
        "humans make smoke on purpose sometimes what is wrong with them",
        "the smoke is thick today i stay on the roof and do not fly far",
        "a world without smoke would be better for birds and seeds",
    ]
    return clean_response(pick(patterns))


def confused_clock():
    patterns = [
        "the clock goes tick tock tick tock never stopping",
        "why do humans look at circles on walls to know things",
        "the big hand and the little hand confusing why two hands",
        "if i stare at the clock long enough does it give me seeds",
        "the tower clock rings every hour loud and booming",
        "humans live by the clock i live by the sun simple",
        "my clock is the temple bells and the traffic sounds",
        "the numbers on the clock mean nothing to a pigeon",
        "you check the clock so often but the seeds still come when they come",
        "time is measured by the clock humans say but the sun measures it better",
    ]
    return clean_response(pick(patterns))


def confused_credit_card():
    patterns = [
        "credit card is a piece of plastic humans use to get things",
        "you give the plastic to a human and get things in return strange",
        "i tried to eat a credit card once very hard not food",
        "the plastic rectangle is like money but also not money",
        "your card was declined what does that mean is it sick",
        "debt is when humans owe things to other humans confusing",
        "you humans trade plastic and paper for food but food is free",
        "a credit score sounds like something made up by humans",
        "when i want something i fly to it i do not need a card",
        "the machine ate your card it must be hungry like a pigeon",
    ]
    return clean_response(pick(patterns))
def perch_favorite():
    patterns = [
        f"my favorite perch is the {pick(PERCH_SPOTS)} the view is best there",
        "i have a special spot where the sun hits just right",
        "the best perch is the one with the most seeds nearby",
        f"i sit on the {pick(PERCH_SPOTS)} every morning and watch the city wake",
        "a good perch needs good sun and good safety",
        "i have many perches but one favorite above all others",
        f"the {pick(PERCH_SPOTS)} is mine no other bird sits there",
        "my perch my spot my place in the world",
    ]
    return clean_response(pick(patterns))


def perch_new():
    patterns = [
        f"found a new spot on the {pick(PERCH_SPOTS)} it is nice and warm",
        "exploring the roof found a new perch location",
        f"the new spot near {pick(CITY_SOUNDS)} is interesting different view",
        "a new perch means a new perspective on the world",
        f"i tried the {pick(PERCH_SPOTS)} today different from my usual spot",
        "new perch new adventures new things to see",
        f"the {pick(BIRD_FRIENDS)} showed me a new spot on the {pick(PERCH_SPOTS)}",
        "moving to a new spot keeps life interesting for a pigeon",
    ]
    return clean_response(pick(patterns))


def perch_sun():
    patterns = [
        "basking in the warm sun on my favorite perch perfect",
        "the sun feels so good on my feathers warm and soothing",
        "spreading my wings in the sun letting the warmth in",
        f"sunning myself on the {pick(PERCH_SPOTS)} life is good",
        "the sun hits this spot just right warm pigeon happy pigeon",
        "feeling the warmth go through my feathers to my skin",
        "sunbathing is a pigeons favorite activity after eating",
        "the sunny spot on the ledge is the best spot in the world",
    ]
    return clean_response(pick(patterns))


def perch_view():
    patterns = [
        f"from the {pick(PERCH_SPOTS)} i can see the whole city",
        "the view from up here is amazing i see everything",
        f"i watch the {pick(CITY_SOUNDS)} from my spot never gets boring",
        "the city stretches out below me like a blanket of lights",
        "i see the trains moving the cars crawling the people walking",
        "from up here the world looks different more peaceful",
        "my perch has the best view on the whole roof i am lucky",
        "watching the sunset from my perch is the best part of the day",
    ]
    return clean_response(pick(patterns))


def happy():
    patterns = [
        "i am happy today the sun is out and seeds are plenty",
        "cooing happily on my perch life is simple and good",
        "a full belly and a warm sun make for a happy pigeon",
        "happy happy happy seeds and sun and safety",
        "why am i happy because i am a pigeon and i have seeds",
        "joy is a warm spot and a pile of seeds that is enough",
        "i feel good today no cats no noise just peace and seeds",
        "happiness is a simple thing when you are a pigeon",
    ]
    return clean_response(pick(patterns))


def curious():
    patterns = [
        "what is that thing over there i must go look at it",
        "curious about the shiny thing on the ground investigating now",
        f"i see something strange near the {pick(PERCH_SPOTS)} i will check it",
        "a pigeon must be curious to find the best seeds",
        "tilting my head looking at this new thing what is it",
        f"the {pick(CITY_SOUNDS)} is different today i wonder why",
        "curiosity leads to seeds sometimes i must explore",
        "i hop closer to examine the unknown object carefully",
    ]
    return clean_response(pick(patterns))


def scared():
    patterns = [
        f"the {pick(THREATS)} scared me i am shaking a little",
        "scared scared scared i need to find a safe spot",
        "that loud noise made me jump my heart is beating fast",
        "i am hiding under the ledge something bad is nearby",
        f"the {pick(CITY_SOUNDS)} scared me i do not like loud things",
        "fear makes me freeze then fly then hide that is the way",
        "i am scared please tell me the danger is gone",
        "something spooked me i need a moment to calm down",
    ]
    return clean_response(pick(patterns))


def proud():
    patterns = [
        "i found the biggest seed today i am the best finder",
        "i built the best nest on the roof very proud of it",
        "i out flew a cat today i am fast and strong",
        "no other pigeon on this roof finds seeds like i do",
        "i taught the young ones how to find seeds good teacher",
        "i am proud to be a pigeon on this roof it is my home",
        "i survived another day in the city that is something to be proud of",
        "my flock relies on me and i never let them down",
    ]
    return clean_response(pick(patterns))
TOPICS = [
    ("food_found", food_found),
    ("food_searching", food_searching),
    ("food_bread", food_bread),
    ("food_rice", food_rice),
    ("food_trash", food_trash),
    ("food_seed_preference", food_seed_preference),
    ("food_water", food_water),
    ("rain_shelter", rain_shelter),
    ("rain_enjoy", rain_enjoy),
    ("rain_wet", rain_wet),
    ("rain_puddle", rain_puddle),
    ("weather_hot", weather_hot),
    ("weather_cold", weather_cold),
    ("storm_fear", storm_fear),
    ("threat_cat_spotted", threat_cat_spotted),
    ("threat_cat_chase", threat_cat_chase),
    ("threat_cat_escape", threat_cat_escape),
    ("threat_kite", threat_kite),
    ("threat_dog", threat_dog),
    ("threat_monkey", threat_monkey),
    ("threat_snake", threat_snake),
    ("flight_soaring", flight_soaring),
    ("flight_short", flight_short),
    ("flight_tired", flight_tired),
    ("flight_wind", flight_wind),
    ("flight_gliding", flight_gliding),
    ("flight_sunset", flight_sunset),
    ("nest_material", nest_material),
    ("nest_building", nest_building),
    ("nest_spot", nest_spot),
    ("nest_chicks", nest_chicks),
    ("nest_cozy", nest_cozy),
    ("nest_guarding", nest_guarding),
    ("nest_sunrise", nest_sunrise),
    ("sleep_night", sleep_night),
    ("sleep_dawn", sleep_dawn),
    ("sleep_nap", sleep_nap),
    ("sleep_safe", sleep_safe),
    ("sleep_interrupted", sleep_interrupted),
    ("sleep_dream", sleep_dream),
    ("greet_morning", greet_morning),
    ("greet_evening", greet_evening),
    ("greet_friend", greet_friend),
    ("greet_stranger", greet_stranger),
    ("greet_human", greet_human),
    ("city_cars", city_cars),
    ("city_construction", city_construction),
    ("city_horns", city_horns),
    ("city_sirens", city_sirens),
    ("city_music", city_music),
    ("city_crowd", city_crowd),
    ("city_temple_bells", city_temple_bells),
    ("bird_crow", bird_crow),
    ("bird_sparrow", bird_sparrow),
    ("bird_myna", bird_myna),
    ("bird_parrot", bird_parrot),
    ("bird_kite", bird_kite),
    ("bird_flock", bird_flock),
    ("confused_money", confused_money),
    ("confused_phone", confused_phone),
    ("confused_internet", confused_internet),
    ("confused_politics", confused_politics),
    ("confused_jobs", confused_jobs),
    ("confused_tv", confused_tv),
    ("confused_books", confused_books),
    ("confused_wallet", confused_wallet),
    ("confused_keys", confused_keys),
    ("confused_school", confused_school),
    ("confused_car", confused_car),
    ("confused_shoes", confused_shoes),
    ("confused_watch", confused_watch),
    ("confused_camera", confused_camera),
    ("confused_computer", confused_computer),
    ("confused_paper", confused_paper),
    ("confused_coffee", confused_coffee),
    ("confused_umbrella", confused_umbrella),
    ("confused_alarm", confused_alarm),
    ("confused_smoke", confused_smoke),
    ("confused_clock", confused_clock),
    ("confused_credit_card", confused_credit_card),
    ("perch_favorite", perch_favorite),
    ("perch_new", perch_new),
    ("perch_sun", perch_sun),
    ("perch_view", perch_view),
    ("happy", happy),
    ("curious", curious),
    ("scared", scared),
    ("proud", proud),
]
CATEGORIES = {
    "food_found": "food", "food_searching": "food", "food_bread": "food",
    "food_rice": "food", "food_trash": "food", "food_seed_preference": "food", "food_water": "food",
    "rain_shelter": "rain", "rain_enjoy": "rain", "rain_wet": "rain", "rain_puddle": "rain",
    "weather_hot": "weather", "weather_cold": "weather", "storm_fear": "weather",
    "threat_cat_spotted": "threats", "threat_cat_chase": "threats", "threat_cat_escape": "threats",
    "threat_kite": "threats", "threat_dog": "threats", "threat_monkey": "threats", "threat_snake": "threats",
    "flight_soaring": "flight", "flight_short": "flight", "flight_tired": "flight",
    "flight_wind": "flight", "flight_gliding": "flight", "flight_sunset": "flight",
    "nest_material": "nesting", "nest_building": "nesting", "nest_spot": "nesting",
    "nest_chicks": "nesting", "nest_cozy": "nesting", "nest_guarding": "nesting", "nest_sunrise": "nesting",
    "sleep_night": "sleep", "sleep_dawn": "sleep", "sleep_nap": "sleep",
    "sleep_safe": "sleep", "sleep_interrupted": "sleep", "sleep_dream": "sleep",
    "greet_morning": "greetings", "greet_evening": "greetings", "greet_friend": "greetings",
    "greet_stranger": "greetings", "greet_human": "greetings",
    "city_cars": "city_sounds", "city_construction": "city_sounds", "city_horns": "city_sounds",
    "city_sirens": "city_sounds", "city_music": "city_sounds", "city_crowd": "city_sounds",
    "city_temple_bells": "city_sounds",
    "bird_crow": "birds", "bird_sparrow": "birds", "bird_myna": "birds",
    "bird_parrot": "birds", "bird_kite": "birds", "bird_flock": "birds",
    "confused_money": "confused", "confused_phone": "confused", "confused_internet": "confused",
    "confused_politics": "confused", "confused_jobs": "confused", "confused_tv": "confused",
    "confused_books": "confused", "confused_wallet": "confused", "confused_keys": "confused",
    "confused_school": "confused", "confused_car": "confused", "confused_shoes": "confused",
    "confused_watch": "confused", "confused_camera": "confused", "confused_computer": "confused",
    "confused_paper": "confused", "confused_coffee": "confused", "confused_umbrella": "confused",
    "confused_alarm": "confused", "confused_smoke": "confused", "confused_clock": "confused",
    "confused_credit_card": "confused",
    "perch_favorite": "perch", "perch_new": "perch", "perch_sun": "perch", "perch_view": "perch",
    "happy": "emotions", "curious": "emotions", "scared": "emotions", "proud": "emotions",
}


PREFIXES = [
    "", "tell me ", "hey pigeon ", "listen bird ", "so ", "i wonder ", "little bird ", "hey ", "do you know ", "excuse me ", "say ", "quick question ", "my friend ", "dear pigeon ", "hello ", "greetings ",
    "tell me pigeon ", "yo pigeon ", "please explain ", "pigeon tell me ", "bird friend ", "tell us ", "just tell me ", "i want to know ", "can you say ", "answer me ", "hey there ", "dear bird ", "tell me now ", "listen to me ", "feathered one "
]
SUFFIXES = [
    "", " right", " what do you think", " huh", " tell me", " please", " any thoughts", " if you know", " maybe", " yes", " what say you", " hmm", " let me know", " i wonder", " bird", " now",
    " today", " my friend", " if you please", " or not", " dear bird", " coo friend", " tell me please", " i beg you", " quickly", " immediately", " if you can", " truly", " indeed", " right now", " my bird friend"
]

def make_question_dynamic(question):
    words = question.split()
    new_words = []
    for w in words:
        w_clean = w.lower().strip("?.!,")
        suffix = w[len(w_clean):]  # keep punctuation
        
        if w_clean in ("seeds", "seed"):
            replacement = pick(SEED_TYPES)
        elif w_clean == "bread":
            replacement = pick(["bread", "roti", "chapati", "naan", "crust"])
        elif w_clean == "rice":
            replacement = pick(["rice", "basmati", "cooked rice", "rice grains"])
        elif w_clean in ("trash", "garbage"):
            replacement = pick(["trash", "garbage", "dumpster", "bin", "waste"])
        elif w_clean == "water":
            replacement = pick(["water", "rainwater", "puddle", "drain water"])
        elif w_clean == "cat":
            replacement = pick(["cat", "stray cat", "street cat"])
        elif w_clean == "cats":
            replacement = pick(["cats", "stray cats", "street cats"])
        elif w_clean == "dog":
            replacement = pick(["dog", "stray dog", "street dog"])
        elif w_clean == "dogs":
            replacement = pick(["dogs", "stray dogs", "street dogs"])
        elif w_clean == "monkey":
            replacement = pick(["monkey", "langur", "macaque"])
        elif w_clean == "monkeys":
            replacement = pick(["monkeys", "langurs", "macaques"])
        elif w_clean == "snake":
            replacement = pick(["snake", "cobra", "viper"])
        elif w_clean == "snakes":
            replacement = pick(["snakes", "cobras", "vipers"])
        elif w_clean == "kite":
            replacement = pick(["kite", "hawk", "eagle"])
        elif w_clean == "kites":
            replacement = pick(["kites", "hawks", "eagles"])
        elif w_clean == "wind":
            replacement = pick(["wind", "strong wind", "breeze"])
        elif w_clean == "rain":
            replacement = pick(["rain", "drizzle", "downpour", "monsoon"])
        elif w_clean == "storm":
            replacement = pick(["storm", "thunderstorm", "dust storm"])
        elif w_clean in ("roof", "rooftop"):
            replacement = pick(PERCH_SPOTS)
        elif w_clean == "nest":
            replacement = pick(["nest", "cozy nest", "twig nest"])
        elif w_clean == "morning":
            replacement = pick(["morning", "dawn", "sunrise"])
        elif w_clean == "evening":
            replacement = pick(["evening", "dusk", "sunset"])
        elif w_clean in ("siren", "sirens"):
            replacement = pick(["siren", "ambulance siren", "police siren"])
        elif w_clean in ("horn", "horns"):
            replacement = pick(["honk", "rickshaw honk", "car horn"])
        elif w_clean == "pigeon":
            replacement = pick(["pigeon", "dove", "rooftop bird"])
        elif w_clean == "pigeons":
            replacement = pick(["pigeons", "doves", "rooftop birds"])
        elif w_clean == "bird":
            replacement = pick(BIRD_FRIENDS + ["bird", "feathered friend"])
        elif w_clean == "birds":
            replacement = pick([b + "s" for b in BIRD_FRIENDS] + ["birds", "feathered friends"])
        elif w_clean == "city":
            replacement = pick(["city", "town", "neighborhood"])
        elif w_clean == "street":
            replacement = pick(["street", "alley", "road"])
        elif w_clean == "streets":
            replacement = pick(["streets", "alleys", "roads"])
        elif w_clean == "human":
            replacement = pick(HUMAN_TYPES + ["human", "person", "seed giver"])
        elif w_clean == "humans":
            replacement = pick([h + "s" if not h.endswith("y") else h[:-1] + "ies" for h in HUMAN_TYPES] + ["humans", "people", "city dwellers"])
        elif w_clean == "animal":
            replacement = pick(CITY_ANIMALS + ["animal", "creature"])
        elif w_clean == "animals":
            replacement = pick([a + "s" for a in CITY_ANIMALS] + ["animals", "creatures"])
        elif w_clean == "food":
            replacement = pick(["food", "grains", "feed"])
        elif w_clean == "hungry":
            replacement = pick(["hungry", "peckish", "starving"])
        elif w_clean == "sky":
            replacement = pick(["sky", "open air", "blue above"])
        elif w_clean == "home":
            replacement = pick(["home", "nesting spot"])
        elif w_clean == "building":
            replacement = pick(["building", "high-rise", "tower"])
        elif w_clean == "buildings":
            replacement = pick(["buildings", "towers", "high-rises"])
        else:
            replacement = w_clean
            
        new_words.append(replacement + suffix)
        
    final_q = " ".join(new_words)
    p = pick(PREFIXES)
    s = pick(SUFFIXES)
    return p + final_q + s


def generate_dataset(n=500000):
    # Build category-balanced weights so each category gets equal share
    cat_to_topics = {}
    for topic_name, topic_fn in TOPICS:
        cat = CATEGORIES[topic_name]
        cat_to_topics.setdefault(cat, []).append((topic_name, topic_fn))
    n_categories = len(cat_to_topics)
    topic_weights = []
    for topic_name, _ in TOPICS:
        cat = CATEGORIES[topic_name]
        n_topics_in_cat = len(cat_to_topics[cat])
        topic_weights.append(1.0 / (n_categories * n_topics_in_cat))

    samples = []
    seen_inputs = set()
    for _ in range(n):
        attempts = 0
        while attempts < 100:
            topic_name, topic_fn = random.choices(TOPICS, weights=topic_weights, k=1)[0]
            question = pick(QUESTIONS[topic_name])
            question = make_question_dynamic(question)
            if question not in seen_inputs:
                seen_inputs.add(question)
                break
            attempts += 1

        response = add_fluff(topic_fn())
        category = CATEGORIES[topic_name]
        samples.append({
            "input": question,
            "output": response,
            "category": category,
        })
    return samples


def main():
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500000
    print(f"generating {n} samples...")
    data = generate_dataset(n)

    from collections import Counter
    cat_counts = Counter(s["category"] for s in data)
    total = len(data)

    with open("pigeon_data.jsonl", "w", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"saved {total} samples to pigeon_data.jsonl")
    print(f"categories ({len(cat_counts)}):")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count} ({count/total*100:.1f}%)")


if __name__ == "__main__":
    main()
