"""
Semantic Analysis Module for Project Analysis Tool

This module provides NLP capabilities for extracting entities, relationships,
and semantic meaning from technical text.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Enumeration of entity types for semantic analysis."""

    TECHNOLOGY = "technology"
    MATERIAL = "material"
    COMPANY = "company"
    ORGANIZATION = "organization"
    PROCESS = "process"
    METHODOLOGY = "methodology"
    RESEARCH_INSTITUTION = "research_institution"
    EQUIPMENT = "equipment"
    MACHINERY = "machinery"
    PRODUCT = "product"
    STANDARD = "standard"
    PERSON = "person"
    LOCATION = "location"
    CONCEPT = "concept"
    UNKNOWN = "unknown"


class SentimentType(Enum):
    """Enumeration of sentiment types for text analysis."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class RelationshipType(Enum):
    """Enumeration of relationship types between entities."""

    DEVELOPS = "develops"
    USES = "uses"
    PRODUCEs = "produces"
    APPLIES = "applies"
    RESEARCHES = "researches"
    PARTNERS = "partners"
    COMPETES = "competes"
    SUPPLIES = "supplies"
    IMPLEMENTS = "implements"
    IMPROVES = "improves"
    REPLACES = "replaces"
    DEPENDS_ON = "depends_on"
    REFERENCES = "references"
    CONTAINS = "contains"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """Represents a named entity extracted from text."""

    text: str
    type: EntityType
    start_pos: int
    end_pos: int
    confidence: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)
    normalized_form: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "type": self.type.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "properties": self.properties,
            "normalized_form": self.normalized_form,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        return cls(
            text=data["text"],
            type=EntityType(data["type"]),
            start_pos=data["start_pos"],
            end_pos=data["end_pos"],
            confidence=data.get("confidence", 1.0),
            properties=data.get("properties", {}),
            normalized_form=data.get("normalized_form"),
        )


@dataclass
class Relationship:
    """Represents a relationship between two entities."""

    source_entity: Entity
    target_entity: Entity
    relationship_type: RelationshipType
    confidence: float = 1.0
    context: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_entity.text,
            "target": self.target_entity.text,
            "type": self.relationship_type.value,
            "confidence": self.confidence,
            "context": self.context,
            "properties": self.properties,
        }


@dataclass
class SentimentAnalysis:
    """Represents sentiment analysis result for text."""

    sentiment: SentimentType
    score: float
    confidence: float
    aspects: dict[str, SentimentType] = field(default_factory=dict)
    strategic_implications: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentiment": self.sentiment.value,
            "score": self.score,
            "confidence": self.confidence,
            "aspects": {k: v.value for k, v in self.aspects.items()},
            "strategic_implications": self.strategic_implications,
        }


class SemanticAnalyzer:
    """
    Semantic analyzer for extracting entities, relationships, and meaning from text.

    This class provides:
    - Named entity recognition (NER)
    - Entity classification and disambiguation
    - Relationship extraction
    - Sentiment and intent analysis
    - Semantic similarity calculations
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the semantic analyzer.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._initialize_patterns()

    def _initialize_patterns(self) -> None:
        """Initialize regex patterns for entity extraction."""
        self._company_patterns = [
            r"\b[A-Z][a-zA-Z]*(?: Inc\.?|Corp\.?|LLC|Ltd\.?|Group|Industries)\b",
            r"\b[A-Z]{2,}(?: Corporation| Company| Corp\.?)\b",
        ]

        self._technology_patterns = [
            r"\b(?:AI|ML|ML|LLM|API|SDK|IoT|5G|6G|cloud|edge|quantum)\b",
            r"\b[A-Z][a-zA-Z]+(?:OS|Platform|Framework|Engine|Service)\b",
        ]

        self._material_patterns = [
            r"\b(?:aluminum|aluminium|steel|copper|titanium|carbon fiber|composite)\b",
            r"\b(?:polymer|ceramic|glass|metal|alloy)s?\b",
        ]

        self._process_patterns = [
            r"\b(?:extrusion|casting|forging|rolling|machining|welding)\b",
            r"\b(?:annealing|quenching|tempering|heat treatment)\b",
        ]

        self._equipment_patterns = [
            r"\b(?:furnace|press|roller|mold|die|cNC|robot|automation)\b",
            r"\b(?:machine|system|unit|device|apparatus)\b",
        ]

    def analyze_text(self, text: str) -> dict[str, Any]:
        """
        Perform comprehensive semantic analysis on text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary containing entities, relationships, and sentiment analysis
        """
        entities = self.extract_entities(text)
        relationships = self.extract_relationships(entities, text)
        sentiment = self.analyze_sentiment(text)
        intent = self.extract_intent(text)

        return {
            "entities": [e.to_dict() for e in entities],
            "relationships": [r.to_dict() for r in relationships],
            "sentiment": sentiment.to_dict() if sentiment else None,
            "intent": intent,
            "entity_count": len(entities),
            "relationship_count": len(relationships),
        }

    def extract_entities(self, text: str) -> list[Entity]:
        """
        Extract named entities from text.

        Args:
            text: Input text to analyze

        Returns:
            List of extracted entities
        """
        entities = []

        companies = self._extract_entities_by_patterns(
            text, self._company_patterns, EntityType.COMPANY
        )
        entities.extend(companies)

        technologies = self._extract_entities_by_patterns(
            text, self._technology_patterns, EntityType.TECHNOLOGY
        )
        entities.extend(technologies)

        materials = self._extract_entities_by_patterns(
            text, self._material_patterns, EntityType.MATERIAL
        )
        entities.extend(materials)

        processes = self._extract_entities_by_patterns(
            text, self._process_patterns, EntityType.PROCESS
        )
        entities.extend(processes)

        equipment = self._extract_entities_by_patterns(
            text, self._equipment_patterns, EntityType.EQUIPMENT
        )
        entities.extend(equipment)

        deduplicated = self._deduplicate_entities(entities)

        return deduplicated

    def _extract_entities_by_patterns(
        self, text: str, patterns: list[str], entity_type: EntityType
    ) -> list[Entity]:
        """Extract entities matching given patterns."""
        entities = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = Entity(
                    text=match.group(),
                    type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.9,
                )
                entities.append(entity)
        return entities

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """Remove duplicate entities based on text and position."""
        seen = set()
        unique = []
        for entity in entities:
            key = (entity.text.lower(), entity.start_pos)
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        return unique

    def classify_entity(self, entity: Entity, context: str) -> EntityType:
        """
        Classify an entity based on context.

        Args:
            entity: Entity to classify
            context: Surrounding text context

        Returns:
            Classified entity type
        """
        text_lower = entity.text.lower()

        technology_indicators = ["technology", "system", "platform", "framework", "algorithm"]
        material_indicators = ["material", "metal", "alloy", "composite", "polymer"]
        process_indicators = ["process", "method", "technique", "procedure"]
        equipment_indicators = ["machine", "equipment", "device", "tool"]

        context_lower = context.lower()

        for indicator in technology_indicators:
            if indicator in context_lower:
                return EntityType.TECHNOLOGY

        for indicator in material_indicators:
            if indicator in context_lower:
                return EntityType.MATERIAL

        for indicator in process_indicators:
            if indicator in context_lower:
                return EntityType.PROCESS

        for indicator in equipment_indicators:
            if indicator in context_lower:
                return EntityType.EQUIPMENT

        return entity.type

    def normalize_entity(self, entity: Entity) -> str:
        """
        Normalize entity text to a standard form.

        Args:
            entity: Entity to normalize

        Returns:
            Normalized form of entity text
        """
        normalized = entity.text.lower().strip()

        normalizations = {
            "ai": "Artificial Intelligence",
            "ml": "Machine Learning",
            "llm": "Large Language Model",
            "api": "Application Programming Interface",
            "sdk": "Software Development Kit",
            "iot": "Internet of Things",
        }

        if normalized in normalizations:
            return normalizations[normalized]

        return normalized.title()

    def extract_relationships(self, entities: list[Entity], text: str) -> list[Relationship]:
        """
        Extract relationships between entities.

        Args:
            entities: List of extracted entities
            text: Original text for context

        Returns:
            List of extracted relationships
        """
        relationships = []

        relationship_patterns = {
            RelationshipType.DEVELOPS: [
                r"(\w+)\s+(?:developed|creates?|builds?|designs?)\s+(\w+)",
                r"(\w+)\s+(?:researching|working on)\s+(\w+)",
            ],
            RelationshipType.USES: [
                r"(\w+)\s+(?:uses?|utilizes?|employs?)\s+(\w+)",
                r"(\w+)\s+(?:based on|built on|powered by)\s+(\w+)",
            ],
            RelationshipType.PRODUCEs: [
                r"(\w+)\s+(?:produces?|manufactures?|makes?)\s+(\w+)",
                r"(\w+)\s+(?:offering|provides?|supplies?)\s+(\w+)",
            ],
            RelationshipType.IMPROVES: [
                r"(\w+)\s+(?:improves?|enhances?|optimizes?)\s+(\w+)",
                r"(\w+)\s+(?:better than|supersedes?|replaces?)\s+(\w+)",
            ],
        }

        for rel_type, patterns in relationship_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    source_text = match.group(1)
                    target_text = match.group(2)

                    source = self._find_entity_by_text(entities, source_text)
                    target = self._find_entity_by_text(entities, target_text)

                    if source and target:
                        relationship = Relationship(
                            source_entity=source,
                            target_entity=target,
                            relationship_type=rel_type,
                            confidence=0.8,
                            context=text[max(0, match.start() - 50) : match.end() + 50],
                        )
                        relationships.append(relationship)

        return relationships

    def _find_entity_by_text(self, entities: list[Entity], text: str) -> Entity | None:
        """Find entity by text match."""
        text_lower = text.lower()
        for entity in entities:
            if entity.text.lower() == text_lower:
                return entity
        return None

    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """
        Analyze sentiment of text.

        Args:
            text: Input text to analyze

        Returns:
            Sentiment analysis result
        """
        positive_words = [
            "innovative",
            "breakthrough",
            "advanced",
            "efficient",
            "promising",
            "successful",
            "excellent",
            "superior",
            "cutting-edge",
            "revolutionary",
        ]
        negative_words = [
            "problem",
            "challenge",
            "issue",
            "failure",
            "difficult",
            "complex",
            "expensive",
            "limited",
            "inefficient",
            "risky",
            "uncertain",
        ]
        strategic_words = [
            "investment",
            "market",
            "growth",
            "strategy",
            "competitive",
            "adoption",
            "implementation",
            "commercialization",
            "scale",
        ]

        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)

        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)
        strategic_count = sum(1 for w in words if w in strategic_words)

        total = len(words) or 1

        positive_score = positive_count / total
        negative_score = negative_count / total

        if positive_score > negative_score:
            sentiment = SentimentType.POSITIVE
            score = positive_score - negative_score
        elif negative_score > positive_score:
            sentiment = SentimentType.NEGATIVE
            score = negative_score - positive_score
        else:
            sentiment = SentimentType.NEUTRAL
            score = 0.0

        if positive_count > 0 and negative_count > 0:
            sentiment = SentimentType.MIXED

        confidence = min(0.95, (positive_count + negative_count) / 10 + 0.5)

        implications = []
        if strategic_count > 0:
            implications.append("Strategic importance detected in the text")
        if positive_count > negative_count:
            implications.append("Overall positive outlook for the technology")
        if negative_count > positive_count:
            implications.append("Potential challenges or risks identified")

        return SentimentAnalysis(
            sentiment=sentiment,
            score=score,
            confidence=confidence,
            aspects={},
            strategic_implications=implications,
        )

    def extract_intent(self, text: str) -> dict[str, Any]:
        """
        Extract strategic intent from text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary containing intent analysis
        """
        text_lower = text.lower()

        intent_indicators = {
            "announcement": ["announced", "unveiled", "revealed", "introduced"],
            "research": ["study", "research", "investigation", "analysis", "findings"],
            "deployment": ["deployment", "implementation", "adoption", "rollout", "launch"],
            "partnership": ["partnered", "collaboration", "joint", "agreement", "alliance"],
            "investment": ["invested", "funding", "investment", "capital", "round"],
            "acquisition": ["acquired", "acquisition", "merger", "bought"],
        }

        detected_intents = []
        for intent, indicators in intent_indicators.items():
            if any(ind in text_lower for ind in indicators):
                detected_intents.append(intent)

        technology_readiness_keywords = {
            "trl": r"trl\s*(\d)",
            "prototype": r"(prototype|pilot|demonstration)",
            "commercial": r"(commercial|production|market-ready)",
            "research_stage": r"(research|development|concept)",
        }

        readiness_indicators = {}
        for key, pattern in technology_readiness_keywords.items():
            match = re.search(pattern, text_lower)
            if match:
                readiness_indicators[key] = match.group(1) if match.lastindex else True

        return {
            "detected_intents": detected_intents,
            "readiness_indicators": readiness_indicators,
            "has_strategy": "strategy" in text_lower or "roadmap" in text_lower,
            "has_timeline": any(
                w in text_lower for w in ["2024", "2025", "2026", "Q1", "Q2", "Q3", "Q4"]
            ),
        }

    def calculate_similarity(
        self, text1: str, text2: str, model_name: str | None = None
    ) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            model_name: Optional embedding model name

        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(re.findall(r"\b\w+\b", text1.lower()))
        words2 = set(re.findall(r"\b\w+\b", text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        jaccard_similarity = len(intersection) / len(union) if union else 0.0

        important_words1 = {w for w in words1 if len(w) > 3}
        important_words2 = {w for w in words2 if len(w) > 3}

        if important_words1 and important_words2:
            important_intersection = important_words1 & important_words2
            important_similarity = len(important_intersection) / len(
                important_words1 | important_words2
            )
        else:
            important_similarity = 0.0

        combined_score = (jaccard_similarity * 0.4) + (important_similarity * 0.6)

        return min(1.0, combined_score)

    def cluster_entities(
        self, entities: list[Entity], similarity_threshold: float = 0.7
    ) -> list[list[Entity]]:
        """
        Cluster entities based on semantic similarity.

        Args:
            entities: List of entities to cluster
            similarity_threshold: Minimum similarity for clustering

        Returns:
            List of entity clusters
        """
        if not entities:
            return []

        clusters = []
        assigned = set()

        for i, entity1 in enumerate(entities):
            if i in assigned:
                continue

            cluster = [entity1]
            assigned.add(i)

            for j, entity2 in enumerate(entities):
                if j in assigned:
                    continue

                similarity = self.calculate_similarity(entity1.text, entity2.text)

                if similarity >= similarity_threshold:
                    cluster.append(entity2)
                    assigned.add(j)

            clusters.append(cluster)

        return clusters
