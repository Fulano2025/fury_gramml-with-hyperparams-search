
def augment_non_terminal(symbol, option):
    new_option = option    
    if type(option) == list and len(option) > 1:
        new_option = ',SEP,'.join(option).split(',')
    if symbol.endswith('_NODE'):
        new_option = ['NODE_START'] + new_option + ['NODE_END'] 
    elif symbol.endswith('_STEPS'):
        new_option = ['STEPS_START'] + new_option + ['STEPS_END'] 
    elif symbol.endswith('_HYPER'):
        new_option = ['HP_START'] + new_option + ['HP_END'] 
    return new_option   

    
def augment_grammar(grammar, column_selector):
    return {symbol:[augment_non_terminal(symbol, option) for option in options] for symbol, options in dot_symbols_augmentation(
        grammar, column_selector).items()} 


def restrict_grammar(grammar, column_types, grammar_key):
    
    # chequeo si contiene la palabra, por ej. NUMERICAL
    grammar[grammar_key] = [
        option for column_type in column_types 
            for option in grammar[grammar_key] 
                if column_type in ' '.join(option)]
    
    return grammar


def components_to_symbols(components, default_hyperparams = True):

    COMP_SYMBOLS = {}

    for component_name, comp_config  in components.items():

        COMPONENT = component_name.upper()

        if not default_hyperparams and "hyperparams" in comp_config and len(comp_config["hyperparams"]) > 0:

            HP_COMPONENT = f'HP_{COMPONENT}'
            
            #Agrego No-terminal de hiperparámetros para este componente particular
            HP_COMPONENT_CHOICES = f'{HP_COMPONENT}_HYPER'

            COMP_SYMBOLS[f'{COMPONENT}'] = [
                [f'"class":"{comp_config["class"]}"', HP_COMPONENT_CHOICES]
            ]

            COMP_SYMBOLS[HP_COMPONENT_CHOICES] = []

            for hp in comp_config["hyperparams"]:

                HP_NAME = hp["name"].upper()

                COMP_SYMBOLS[HP_COMPONENT_CHOICES].append(f'{HP_COMPONENT}_{HP_NAME}_NODE')
                
                COMP_SYMBOLS[f'{HP_COMPONENT}_{HP_NAME}_NODE'] = [
                    [f'"name":"{hp["name"]}"', f'{HP_COMPONENT}_{HP_NAME}_VALUE']
                ]
                
                COMP_SYMBOLS[f'{HP_COMPONENT}_{HP_NAME}_VALUE'] = []

                if "values" in hp:

                    for ix in range(len(hp["values"])):
                        COMP_SYMBOLS[f'{HP_COMPONENT}_{HP_NAME}_VALUE'].append([
                            f'"value":"{hp["values"][ix]}"' if type(hp["values"][ix])==str 
                                else f'"value":{hp["values"][ix]}'])

                else: # Si no tiene opciones (values) uso el param por defecto (value)

                    COMP_SYMBOLS[f'{HP_COMPONENT}_{HP_NAME}_VALUE'].append(
                            f'"value":"{hp["value"]}"' if type(hp["value"])==str 
                                else f'"value":{hp["value"]}')
                
                    # corrijo para que sea lista de listas
                    COMP_SYMBOLS[f'{HP_COMPONENT}_{HP_NAME}_VALUE'] = [COMP_SYMBOLS[f'{HP_COMPONENT}_{HP_NAME}_VALUE']] 
            
            # corrijo para que sea lista de listas        
            COMP_SYMBOLS[HP_COMPONENT_CHOICES] = [COMP_SYMBOLS[HP_COMPONENT_CHOICES]] 
            
        else:   

            COMP_SYMBOLS[f'{COMPONENT}'] = [
                [str(comp_config)[1:-1]]
            ]
                           
    return COMP_SYMBOLS


def dot_symbols_augmentation(grammar, column_selector):
    for symbol, options in grammar.items():
        new_options = []
        for option in options:
            new_option = []
            for elem in option:
                if elem.endswith(".component"):
                    new_option.append(elem.replace(".component","").upper())
                elif elem.endswith(".columns"):
                    json_element, columns_type, _ = elem.split(".")
                    arr_columns = column_selector.select_columns_by_type(columns_type) #TODO
                    if arr_columns is None or len(arr_columns) == 0:
                        pass #print(f'WARNING: No columns of type {columns_type} founded.')
                    new_option.append(json_element + str(arr_columns))                   
                else:
                    new_option.append(elem)
            new_options.append(new_option)
        grammar[symbol] = new_options
                
    return grammar



def replace_symbol_values(grammar, old, new):
    for symbol, options in grammar.items():
        new_options = []
        for option in options:
            new_option = []
            for elem in option:
                if elem == old:
                    new_option.append(new)                
                else:
                    new_option.append(elem)
            new_options.append(new_option)
        grammar[symbol] = new_options
    return grammar


def has_reducible_symbols(grammar):
    for key, values in grammar.items():
        if len(values) == 1: #symbolos que tienen una sola opcion
            if len(values[0]) > 1 and all([not elem.isupper() for elem in values[0]]):
                return True
            elif len(values[0]) == 1 and not values[0][0].isupper():
                return True
    return False


def reduce_grammar(grammar):
    
    reduced_grammar = grammar.copy()
    
    for key, values in grammar.items():
        if len(values) == 1 and len(values[0]) == 1 and not values[0][0].isupper():
            del reduced_grammar[key]
            reduced_grammar = replace_symbol_values(reduced_grammar, key, values[0][0])

    for key, values in reduced_grammar.items():
        if len(values) == 1 and len(values[0]) > 1 and all([not elem.isupper() for elem in values[0]]):
            reduced_grammar[key] = [[''.join(values[0])]]
            
    return reduced_grammar


def get_minimal_grammar(grammar):
    
    reduced_grammar = grammar.copy()
    
    while has_reducible_symbols(reduced_grammar):
        reduced_grammar = reduce_grammar(reduced_grammar)
        
    return reduced_grammar



def get_number_of_paths(search_space, initial_symbol):

    def get_number_of_path_rule(rule):
        suma = 0
        for option in rule:
            suma += get_number_of_path_option(option)
        return suma

    def get_number_of_path_option(option):
        suma = 1
        for symbol in option:
            suma *= get_number_of_path_symbol(symbol)
        return suma

    def get_number_of_path_symbol(symbol):
        if not symbol.isupper():
            return 1
        else:
            return get_number_of_path_rule(search_space[symbol])
    
    return get_number_of_path_rule(search_space[initial_symbol])
    