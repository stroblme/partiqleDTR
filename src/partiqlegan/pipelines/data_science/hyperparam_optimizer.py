import optuna as o

class Hyperparam_Optimizer():
    def initialize_storage(host:str, port:int, path:str, password:str):
        """
        Storage intialization
        """
        storage = o.storages.redis.RedisStorage(
                    url=f'redis://{password}@{host}:{port}/{path}',
                )

        return storage

