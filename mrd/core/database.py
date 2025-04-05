"""
Database connection module for Market Regime Dashboard using SQLAlchemy.
"""
import pandas as pd
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

# Set up logger
logger = logging.getLogger(__name__)

class Database:
    """Database connection and query execution class using SQLAlchemy."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for database connection."""
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self, config=None):
        """Initialize database connection with SQLAlchemy engine."""
        if self._initialized:
            return
            
        try:
            if config:
                # Use provided config
                db_config = {
                    'host': config.get('database', 'host', 'localhost'),
                    'port': config.get('database', 'port', 5432),
                    'username': config.get('database', 'username', 'postgres'),
                    'password': config.get('database', 'password', 'postgres'),
                    'database_name': config.get('database', 'database_name', 'evolabz')
                }
            else:
                # Use imported config
                from mrd.core.config import config as default_config
                db_config = {
                    'host': default_config.get('database', 'host', 'localhost'),
                    'port': default_config.get('database', 'port', 5432),
                    'username': default_config.get('database', 'username', 'postgres'),
                    'password': default_config.get('database', 'password', 'postgres'),
                    'database_name': default_config.get('database', 'database_name', 'evolabz')
                }
            
            # Construct database URL
            self.url = f"postgresql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database_name']}"
            self.engine = create_engine(self.url)
            
            logger.info(f"Database connector initialized successfully with URL: {self.url}")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize database connector: {e}")
            raise
    
    def connect(self):
        """Test database connection."""
        try:
            with self.engine.connect() as connection:
                logger.info("Successfully connected to the database")
                return True
        except OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            raise
            
    def execute_query(self, query, params=None):
        """
        Execute a SQL query and return results as a pandas DataFrame.
        """
        try:
            # Convert query to SQLAlchemy text object
            sql = text(query)
            
            # Execute query and convert to DataFrame
            with self.engine.connect() as connection:
                result = pd.read_sql(sql, connection, params=params)
                
            logger.info(f"Query executed successfully, returned {len(result)} rows")
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
            
    def execute_update(self, query, params=None):
        """
        Execute an SQL update/insert/delete statement.
        """
        try:
            # Convert query to SQLAlchemy text object
            sql = text(query)
            
            # Execute statement
            with self.engine.connect() as connection:
                transaction = connection.begin()
                try:
                    result = connection.execute(sql, params)
                    affected_rows = result.rowcount
                    transaction.commit()
                except:
                    transaction.rollback()
                    raise
                    
            logger.info(f"Update executed successfully, affected {affected_rows} rows")
            return affected_rows
            
        except Exception as e:
            logger.error(f"Update execution failed: {e}")
            raise

# Singleton instance for easy importing
db = Database()

# Convenience functions
def get_connection():
    """Get the SQLAlchemy engine for direct connection."""
    return db.engine
    
def execute_query(query, params=None):
    """Execute a SQL query and return results as a DataFrame."""
    return db.execute_query(query, params)
    
def execute_update(query, params=None):
    """Execute an SQL update statement."""
    return db.execute_update(query, params)