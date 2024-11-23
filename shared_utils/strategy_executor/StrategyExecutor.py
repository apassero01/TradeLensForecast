import threading
from queue import Queue, Empty


class StrategyExecutor:
    _instance = None
    _lock = threading.RLock()  # Use RLock instead of Lock

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(StrategyExecutor, cls).__new__(cls)
                    cls._instance.execution_queue = Queue()
                    cls._instance.currently_executing = False
                    cls._instance.current_thread = None
                    cls._instance.strategies = {}
                    cls._instance.nested_execution_count = 0  # Tracks nested executions
        return cls._instance

    def execute(self, entity, strategy_request):
        strategy_name = strategy_request.strategy_name

        # Bypass the lock for nested execution within the same thread
        if self.currently_executing and threading.current_thread() == self.current_thread:
            return self._process_request(strategy_name, entity, strategy_request)

        # Acquire the lock for top-level or external execution
        with self._lock:
            if not self.currently_executing:
                self.currently_executing = True
                self.current_thread = threading.current_thread()
                return self._process_request(strategy_name, entity, strategy_request)
            else:
                self.execution_queue.put((strategy_name, entity, strategy_request))

    def _process_request(self, strategy_name, entity, strategy_request):
        result = None  # Initialize a variable to store the result

        try:
            # Increment nested execution counter
            self.nested_execution_count += 1

            strategy_cls = self.strategies.get(strategy_name)
            if not strategy_cls:
                raise ValueError(f"Strategy {strategy_name} is not registered.")

            # Create and execute the strategy
            strategy = strategy_cls(self, strategy_request)
            result = strategy.apply(entity)  # Store the result in the variable
        except Exception as e:
            # Log the exception and re-raise it
            print(f"Error executing strategy {strategy_name}: {e}")
            raise e
        finally:
            # Decrement nested execution counter
            self.nested_execution_count -= 1

            # Reset state only if no nested executions are left
            if self.nested_execution_count == 0:
                with self._lock:  # This will now work because of RLock
                    self.currently_executing = False
                    self.current_thread = None
                    self._try_next()

        # Return the result after cleanup
        return result

    def _try_next(self):
        try:
            next_request = self.execution_queue.get_nowait()
            if next_request:
                # Process the next request
                strategy_name, entity, strategy_request = next_request
                self.execute(entity, strategy_request)
        except Empty:
            pass

    def register_strategy(self, strategy_name, strategy_cls):
        if not hasattr(self, 'strategies'):
            self.strategies = {}
        self.strategies[strategy_name] = strategy_cls

    @classmethod
    def destroy(cls):
        with cls._lock:
            cls._instance = None