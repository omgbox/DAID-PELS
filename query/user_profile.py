"""
User Profile
Stores and retrieves user preferences, facts, and conversation history.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger('bookbot.query.user_profile')


class UserProfile:
    """Manages user preferences and personal facts."""

    def __init__(self, db_manager=None):
        self.db = db_manager
        self._preferences = {}  # In-memory cache
        self._facts = {}        # In-memory cache
        self._initialized = False

    def initialize(self):
        """Initialize the user profile with DB tables."""
        if self._initialized or not self.db:
            return

        try:
            # Create tables if they don't exist
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    value TEXT NOT NULL,
                    sentiment TEXT DEFAULT 'positive',
                    mention_count INTEGER DEFAULT 1,
                    first_mentioned TEXT,
                    last_mentioned TEXT
                )
            """)

            self.db.execute("""
                CREATE TABLE IF NOT EXISTS user_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_type TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    first_mentioned TEXT,
                    last_mentioned TEXT
                )
            """)

            self.db.execute("""
                CREATE TABLE IF NOT EXISTS learned_knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    source TEXT DEFAULT 'conversation',
                    confidence REAL DEFAULT 0.5,
                    timestamp TEXT
                )
            """)

            self._initialized = True
            logger.info("User profile tables initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize user profile tables: {e}")

    def store_preference(self, category: str, value: str, sentiment: str = 'positive'):
        """
        Store a user preference.

        Args:
            category: Preference category (interest, food, color, etc.)
            value: The preference value
            sentiment: positive, negative, or neutral
        """
        if not self._initialized:
            self.initialize()

        now = datetime.now().isoformat()

        # Update in-memory cache
        key = f"{category}:{value}"
        if key in self._preferences:
            self._preferences[key]['mention_count'] += 1
            self._preferences[key]['last_mentioned'] = now
        else:
            self._preferences[key] = {
                'category': category,
                'value': value,
                'sentiment': sentiment,
                'mention_count': 1,
                'first_mentioned': now,
                'last_mentioned': now,
            }

        # Store in database
        if self.db:
            try:
                # Check if preference already exists
                existing = self.db.execute(
                    "SELECT id, mention_count FROM user_preferences "
                    "WHERE category = ? AND value = ?",
                    (category, value)
                )

                if existing:
                    # Update mention count
                    self.db.execute(
                        "UPDATE user_preferences SET mention_count = ?, last_mentioned = ? "
                        "WHERE id = ?",
                        (existing[0][1] + 1, now, existing[0][0])
                    )
                else:
                    # Insert new preference
                    self.db.execute(
                        "INSERT INTO user_preferences (category, value, sentiment, mention_count, first_mentioned, last_mentioned) "
                        "VALUES (?, ?, ?, 1, ?, ?)",
                        (category, value, sentiment, now, now)
                    )

                logger.debug(f"Stored preference: {category} = {value} ({sentiment})")

            except Exception as e:
                logger.warning(f"Failed to store preference: {e}")

    def store_fact(self, fact_type: str, fact_value: str):
        """
        Store a personal fact about the user.

        Args:
            fact_type: Type of fact (name, age, location, etc.)
            fact_value: The fact value
        """
        if not self._initialized:
            self.initialize()

        now = datetime.now().isoformat()

        # Update in-memory cache
        self._facts[fact_type] = {
            'fact_type': fact_type,
            'fact_value': fact_value,
            'first_mentioned': now,
            'last_mentioned': now,
        }

        # Store in database
        if self.db:
            try:
                existing = self.db.execute(
                    "SELECT id FROM user_facts WHERE fact_type = ?",
                    (fact_type,)
                )

                if existing:
                    # Update existing fact
                    self.db.execute(
                        "UPDATE user_facts SET fact_value = ?, last_mentioned = ? WHERE id = ?",
                        (fact_value, now, existing[0][0])
                    )
                else:
                    # Insert new fact
                    self.db.execute(
                        "INSERT INTO user_facts (fact_type, fact_value, first_mentioned, last_mentioned) "
                        "VALUES (?, ?, ?, ?)",
                        (fact_type, fact_value, now, now)
                    )

                logger.debug(f"Stored fact: {fact_type} = {fact_value}")

            except Exception as e:
                logger.warning(f"Failed to store fact: {e}")

    def get_preferences(self, category: str = None) -> List[Dict]:
        """
        Get user preferences.

        Args:
            category: Optional category filter

        Returns:
            List of preference dicts
        """
        if not self._initialized:
            self.initialize()

        if self.db:
            try:
                if category:
                    rows = self.db.execute(
                        "SELECT category, value, sentiment, mention_count "
                        "FROM user_preferences WHERE category = ? "
                        "ORDER BY mention_count DESC",
                        (category,)
                    )
                else:
                    rows = self.db.execute(
                        "SELECT category, value, sentiment, mention_count "
                        "FROM user_preferences ORDER BY mention_count DESC"
                    )

                return [
                    {
                        'category': r[0],
                        'value': r[1],
                        'sentiment': r[2],
                        'mention_count': r[3],
                    }
                    for r in rows
                ]

            except Exception as e:
                logger.warning(f"Failed to get preferences: {e}")

        # Fallback to in-memory cache
        return list(self._preferences.values())

    def get_fact(self, fact_type: str) -> Optional[str]:
        """
        Get a specific user fact.

        Args:
            fact_type: Type of fact to retrieve

        Returns:
            Fact value or None
        """
        if not self._initialized:
            self.initialize()

        if self.db:
            try:
                rows = self.db.execute(
                    "SELECT fact_value FROM user_facts WHERE fact_type = ?",
                    (fact_type,)
                )
                if rows:
                    return rows[0][0]
            except Exception as e:
                logger.warning(f"Failed to get fact: {e}")

        # Fallback to in-memory cache
        if fact_type in self._facts:
            return self._facts[fact_type]['fact_value']

        return None

    def get_all_facts(self) -> Dict[str, str]:
        """Get all user facts as a dict."""
        if not self._initialized:
            self.initialize()

        if self.db:
            try:
                rows = self.db.execute(
                    "SELECT fact_type, fact_value FROM user_facts"
                )
                return {r[0]: r[1] for r in rows}
            except Exception as e:
                logger.warning(f"Failed to get facts: {e}")

        # Fallback to in-memory cache
        return {k: v['fact_value'] for k, v in self._facts.items()}

    def get_user_name(self) -> Optional[str]:
        """Get the user's name if stored."""
        return self.get_fact('name')

    def get_context_summary(self) -> str:
        """Get a summary of known user information for context."""
        parts = []

        name = self.get_user_name()
        if name:
            parts.append(f"User's name: {name}")

        preferences = self.get_preferences()
        if preferences:
            likes = [p['value'] for p in preferences if p['sentiment'] == 'positive'][:5]
            dislikes = [p['value'] for p in preferences if p['sentiment'] == 'negative'][:3]
            if likes:
                parts.append(f"Likes: {', '.join(likes)}")
            if dislikes:
                parts.append(f"Dislikes: {', '.join(dislikes)}")

        facts = self.get_all_facts()
        for fact_type, fact_value in facts.items():
            if fact_type != 'name':  # Already included above
                parts.append(f"{fact_type}: {fact_value}")

        return '; '.join(parts) if parts else ''
