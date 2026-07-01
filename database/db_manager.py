"""
BookBot Database Manager
SQLite connection and schema management.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import DATABASE, DATABASE_PATH

logger = logging.getLogger('bookbot.database')


class DBManager:
    """SQLite database manager for BookBot."""

    def __init__(self, db_path: str = None):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or DATABASE_PATH
        self.conn = None
        self.cursor = None

    def connect(self):
        """Connect to the database."""
        logger.info(f"Connecting to database: {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Set pragmas
        self.cursor.execute(f"PRAGMA journal_mode=WAL")
        self.cursor.execute(f"PRAGMA synchronous=NORMAL")
        self.cursor.execute(f"PRAGMA cache_size={DATABASE['cache_size_kb']}")
        self.cursor.execute(f"PRAGMA temp_store=MEMORY")
        self.cursor.execute(f"PRAGMA mmap_size={DATABASE['mmap_size_bytes']}")
        self.cursor.execute(f"PRAGMA foreign_keys=ON")

        logger.info("Database connected")

    def disconnect(self):
        """Disconnect from the database."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            logger.info("Database disconnected")

    def initialize_schema(self):
        """Initialize the database schema from schema.sql."""
        schema_path = Path(__file__).parent / "schema.sql"
        logger.info(f"Initializing schema from: {schema_path}")

        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()

        # Split by semicolons and execute each statement
        statements = []
        current = []
        for line in schema_sql.split('\n'):
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('--'):
                continue
            # Skip PRAGMA statements (already set in connect())
            if line.startswith('PRAGMA'):
                continue
            current.append(line)
            if line.endswith(';'):
                stmt = ' '.join(current)
                statements.append(stmt)
                current = []

        # Add any remaining statement
        if current:
            statements.append(' '.join(current))

        # Execute each statement
        for statement in statements:
            statement = statement.strip()
            if not statement:
                continue
            try:
                self.cursor.execute(statement)
            except sqlite3.OperationalError as e:
                # Ignore "already exists" errors
                if "already exists" not in str(e):
                    logger.warning(f"Schema statement failed: {e}")

        self.conn.commit()
        logger.info("Schema initialized")

    def execute(self, sql: str, params: tuple = None) -> List[Tuple]:
        """
        Execute a SQL query.

        Args:
            sql: SQL query
            params: Query parameters

        Returns:
            Query results
        """
        try:
            if params:
                self.cursor.execute(sql, params)
            else:
                self.cursor.execute(sql)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"SQL error: {e}")
            logger.error(f"SQL: {sql}")
            logger.error(f"Params: {params}")
            raise

    def execute_many(self, sql: str, params_list: List[tuple]):
        """
        Execute a SQL query with multiple parameter sets.

        Args:
            sql: SQL query
            params_list: List of parameter tuples
        """
        try:
            self.cursor.executemany(sql, params_list)
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"SQL error: {e}")
            logger.error(f"SQL: {sql}")
            raise

    def insert(self, table: str, data: Dict[str, Any]) -> int:
        """
        Insert a row into a table.

        Args:
            table: Table name
            data: Column-value pairs

        Returns:
            Row ID of inserted row
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        self.cursor.execute(sql, tuple(data.values()))
        self.conn.commit()
        return self.cursor.lastrowid

    def insert_many(self, table: str, data_list: List[Dict[str, Any]]):
        """
        Insert multiple rows into a table.

        Args:
            table: Table name
            data_list: List of column-value dicts
        """
        if not data_list:
            return

        columns = ', '.join(data_list[0].keys())
        placeholders = ', '.join(['?' for _ in data_list[0]])
        sql = f"INSERT OR IGNORE INTO {table} ({columns}) VALUES ({placeholders})"

        params_list = [tuple(d.values()) for d in data_list]
        self.cursor.executemany(sql, params_list)
        self.conn.commit()

    def update(self, table: str, data: Dict[str, Any], where: str, where_params: tuple):
        """
        Update rows in a table.

        Args:
            table: Table name
            data: Column-value pairs to update
            where: WHERE clause
            where_params: WHERE clause parameters
        """
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        params = tuple(data.values()) + where_params
        self.cursor.execute(sql, params)
        self.conn.commit()

    def delete(self, table: str, where: str, where_params: tuple):
        """
        Delete rows from a table.

        Args:
            table: Table name
            where: WHERE clause
            where_params: WHERE clause parameters
        """
        sql = f"DELETE FROM {table} WHERE {where}"
        self.cursor.execute(sql, where_params)
        self.conn.commit()

    def select(self, table: str, columns: str = '*', where: str = None,
               where_params: tuple = None, order_by: str = None,
               limit: int = None) -> List[Tuple]:
        """
        Select rows from a table.

        Args:
            table: Table name
            columns: Columns to select
            where: WHERE clause
            where_params: WHERE clause parameters
            order_by: ORDER BY clause
            limit: LIMIT clause

        Returns:
            Query results
        """
        sql = f"SELECT {columns} FROM {table}"
        if where:
            sql += f" WHERE {where}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit:
            sql += f" LIMIT {limit}"

        return self.execute(sql, where_params)

    def count(self, table: str, where: str = None, where_params: tuple = None) -> int:
        """
        Count rows in a table.

        Args:
            table: Table name
            where: WHERE clause
            where_params: WHERE clause parameters

        Returns:
            Row count
        """
        sql = f"SELECT COUNT(*) FROM {table}"
        if where:
            sql += f" WHERE {where}"

        result = self.execute(sql, where_params)
        return result[0][0] if result else 0

    def get_stats(self) -> Dict[str, int]:
        """
        Get database statistics.

        Returns:
            Dictionary of table names and row counts
        """
        tables = [
            'definitions', 'vocabulary', 'sentences', 'sentence_tokens',
            'word_definition_links', 'entities', 'entity_mentions',
            'svo_triples', 'knowledge_edges', 'coreference_chains',
            'coreference_mentions', 'idiom_lexicon', 'idiom_instances',
            'metaphors', 'topics', 'topic_sentences', 'semantic_fields',
            'temporal_events', 'temporal_order', 'causal_chains',
            'narrative_structure', 'pragmatic_rules', 'ocr_corrections'
        ]

        stats = {}
        for table in tables:
            try:
                stats[table] = self.count(table)
            except:
                stats[table] = 0

        return stats

    def clear_all(self):
        """Clear all data from all tables."""
        tables = [
            'pragmatic_rules', 'narrative_structure', 'causal_chains',
            'temporal_order', 'temporal_events', 'semantic_fields',
            'topic_sentences', 'topics', 'metaphors', 'idiom_instances',
            'idiom_lexicon', 'coreference_mentions', 'coreference_chains',
            'knowledge_edges', 'svo_triples', 'entity_mentions', 'entities',
            'ocr_corrections', 'word_definition_links', 'sentence_tokens',
            'sentences_fts', 'sentences', 'vocabulary', 'definitions',
            'document_stats', 'convergence_log', 'training_metadata'
        ]

        for table in tables:
            try:
                self.cursor.execute(f"DELETE FROM {table}")
            except:
                pass

        self.conn.commit()
        logger.info("All tables cleared")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
