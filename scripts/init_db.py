"""
Database Initialization Script

Sets up the PostgreSQL database with pgvector extension and creates
the necessary tables for storing document chunks and embeddings.
"""

import psycopg2
from pgvector.psycopg2 import register_vector

from finbot.config import DB_URI


def initialize_database():
    """
    Initialize the database schema for FinBot.
    
    Creates:
    - pgvector extension
    - documents table with vector column
    - Index for efficient similarity search
    
    Raises:
        psycopg2.Error: If database setup fails
    """
    try:
        connection = psycopg2.connect(DB_URI)
        register_vector(connection)
        cursor = connection.cursor()
        
        # Create pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                source TEXT NOT NULL,
                chunk TEXT NOT NULL,
                metadata JSONB DEFAULT '{}',
                embedding vector(384)
            );
        """)
        
        # Create index for efficient similarity search
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding 
            ON documents USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
        """)
        
        connection.commit()
        cursor.close()
        connection.close()
        
        print("Database initialized successfully")
        print("- pgvector extension created")
        print("- documents table created")
        print("- similarity search index created")
        
    except psycopg2.Error as e:
        print(f"Database initialization failed: {e}")
        raise


def main():
    """Main entry point for database initialization."""
    try:
        initialize_database()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()