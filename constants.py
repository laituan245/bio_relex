from os.path import dirname, join, realpath

# Basic Constants
BASE_PATH = dirname(realpath(__file__))
BASIC_CONF_PATH = join(BASE_PATH, 'configs/basic.conf')
PRETRAINED_MODEL = None
KNOWLEDGE_MODULE_LOG_FILE = 'knowledge_module_logs.txt'

# Datasets Constants
NOT_ENTITY = 'not-entity'
NOT_RELATION = 'not-relation'

ADE = 'ade'
ADE_ENTITY_TYPES = [NOT_ENTITY, 'Adverse-Effect', 'Drug']
ADE_RELATION_TYPES = [NOT_RELATION, 'Adverse-Effect']

BIORELEX = 'biorelex'
BIORELEX_ENTITY_TYPES = [NOT_ENTITY, 'protein', 'protein-family', 'chemical', 'DNA',
                         'protein-complex', 'protein-domain', 'cell', 'experimental-construct',
                         'RNA', 'experiment-tag', 'reagent', 'protein-motif', 'gene',
                         'amino-acid', 'protein-region', 'assay', 'organelle',
                         'peptide', 'fusion-protein', 'protein-isoform', 'process',
                         'mutation', 'protein-RNA-complex', 'drug', 'organism',
                         'disease', 'protein-DNA-complex', 'brand', 'tissue',
                         'RNA-family', 'gene-family', 'fusion-gene', 'parameter']
BIORELEX_RELATION_TYPES = [NOT_RELATION, -1, 0, 1]

DATASETS = [ADE, BIORELEX]

# Model Save Path
BASE_SAVE_PATH = '/shared/nas/data/m1/tuanml2/tmp'

# Caches
CACHE_DIR = join(BASE_PATH, 'caches')
UMLS_CONCEPTS_SQLITE = join(CACHE_DIR, 'umls_concepts.sqlite')

# MetaMap and UMLS
UMLS_EMBS = join(BASE_PATH, 'resources/umls_embs.pkl')
UMLS_SEMTYPES_FILE = join(BASE_PATH, 'resources/umls_semtypes.txt')
UMLS_RELTYPES_FILE = join(BASE_PATH, 'resources/umls_reltypes.txt')
UMLS_TEXT2GRAPH_FILE = join(BASE_PATH, 'resources/text2graph.pkl')
UMLS_EMBS_SIZE = 50 # https://github.com/r-mal/umls-embeddings
METAMAP_PATH = '/shared/nas/data/m1/tuanml2/software/public_mm/bin/metamap20'
MM_TYPES = ['aapp', 'acab', 'acty', 'aggp', 'amas', 'amph', 'anab', 'anim',
            'anst', 'antb', 'arch', 'bacs', 'bact', 'bdsu', 'bdsy', 'bhvr',
            'biof', 'bird', 'blor', 'bmod', 'bodm', 'bpoc', 'bsoj', 'celc',
            'celf', 'cell', 'cgab', 'chem', 'chvf', 'chvs', 'clas', 'clna',
            'clnd', 'cnce', 'comd', 'crbs', 'diap', 'dora', 'drdd', 'dsyn',
            'edac', 'eehu', 'elii', 'emod', 'emst', 'enty', 'enzy', 'euka',
            'evnt', 'famg', 'ffas', 'fish', 'fndg', 'fngs', 'food', 'ftcn',
            'genf', 'geoa', 'gngm', 'gora', 'grpa', 'grup', 'hcpp', 'hcro',
            'hlca', 'hops', 'horm', 'humn', 'idcn', 'imft', 'inbe', 'inch',
            'inpo', 'inpr', 'irda', 'lang', 'lbpr', 'lbtr', 'mamm', 'mbrt',
            'mcha', 'medd', 'menp', 'mnob', 'mobd', 'moft', 'mosq', 'neop',
            'nnon', 'npop', 'nusq', 'ocac', 'ocdi', 'orch', 'orga', 'orgf',
            'orgm', 'orgt', 'ortf', 'patf', 'phob', 'phpr', 'phsf', 'phsu',
            'plnt', 'podg', 'popg', 'prog', 'pros', 'qlco', 'qnco', 'rcpt',
            'rept', 'resa', 'resd', 'rnlw', 'sbst', 'shro', 'socb', 'sosy',
            'spco', 'tisu', 'tmco', 'topp', 'virs', 'vita', 'vtbt']

# Constants for Mentions Filtering
BIORELEX_FILTER_WORDSET_1 = ['such', 'is', 'et', 'same', 'fact', 'not', 'or', 'were', 'about',
                            'both', 'more', 'al', 'J.', 'M.', 'but', 'has', 'then', 'after',
                            'all', 'can', 'may', 'there', 'have', 'late', 'each']

BIORELEX_FILTER_WORDSET_2 = ['present', 'while', 'constant', 'apparent', 'hypothesis', 'competition',
                             'therefore', 'results', 'would', 'help', 'reaching', 'shown', 'designated',
                             'versus', 'finally', 'and an', 'submit', 'from', 'because', 'regardless',
                             'which', 'together', 'contribute', 'indicating', 'contrast', 'however',
                             'recent', 'these', 'their', 'when', 'reported', 'furthermore', 'ough',
                             'competitively', 'we', 'although']

BIORELEX_FILTER_WORDSET_3 = ['also', 'reveals', 'where', ' that', 'that ', 'directly',
                             'Fig', ' is ', ' the ', ' but ', 'very', 'important',
                             ' not ', 'only', 'was', 'example', 'could', ' can ',
                             'middle', 'publish', 'et al', 'suggest', '199', 'early']

ADE_FILTER_WORDSET_1 = ['after', 'present', 'report', 'reported', 'authors', 'observed',
                        'author', 'developed', 'indicates', 'indicated', 'year'
                        'hours', 'administration', 'probably', 'we', 'duration',
                        'complicated', 'presented', 'is', 'describe', 'old', 'observe',
                        'unclear', 'following', 'remains', 'often', 'used', 'may',
                        'management', 'amounts', 'were', 'minutes', 'hours',
                        'closely', 'despite', 'indian', 'receiving', 'who', 'by',
                        'however', 'detailed', 'purpose', 'was', 'highlight',
                        'starting', 'although', 'few', 'background', 'described',
                        'use', 'completed', 'over', 'conclusion', 'conclusions',
                        'risk', 'patient', 'patients', 'are', 'status', 'presenting',
                        'months', 'years', 'evidence']

ADE_FILTER_WORDSET_2 = ['and', 'this', 'that', 'the', 'in', 'of', 'with', 'from',
                        'like', 'during', '(', ')', ':', 'to']

NODE = 'node'
