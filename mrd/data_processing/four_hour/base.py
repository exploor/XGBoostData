from abc import ABC, abstractmethod

class FeatureCalculator(ABC):
    @abstractmethod
    def calculate(self, df):
        pass