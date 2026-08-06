import json


from .decorators import singleton


@singleton
class ConfigLoader:
    def __init__(self,configPath=r'config.json') -> None:
        with open(configPath) as f:
            self.__config = json.load(f)

    
    def __getitem__(self, item):
        return self.__config[item]
    
if __name__=='__main__':
    a=ConfigLoader()
    b=ConfigLoader()
    print(id(a),id(b))