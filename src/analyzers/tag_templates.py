"""Comprehensive tag templates and taxonomy for video content classification."""

from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class TagCategory(Enum):
    """Main tag categories for video content."""
    SCENE = "scene"
    OBJECT = "object"
    PERSON = "person"
    ACTIVITY = "activity"
    EMOTION = "emotion"
    TECHNICAL = "technical"
    COLOR = "color"
    LIGHTING = "lighting"
    COMPOSITION = "composition"
    AUDIO = "audio"
    STYLE = "style"
    GENRE = "genre"


@dataclass
class TagDefinition:
    """Definition of a single tag with metadata."""
    name: str
    category: TagCategory
    aliases: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    confidence_threshold: float = 0.5
    description: str = ""


class TagTaxonomy:
    """
    Comprehensive tag taxonomy for video content classification.

    Provides hierarchical tag definitions, synonyms, and inference rules.
    """

    # Scene/Location Tags
    SCENE_TAGS: Dict[str, List[str]] = {
        # Indoor locations
        "indoor": ["inside", "interior"],
        "indoor/home": ["house", "apartment", "residence", "living_space"],
        "indoor/home/living_room": ["lounge", "sitting_room", "family_room"],
        "indoor/home/bedroom": ["sleeping_room", "master_bedroom"],
        "indoor/home/kitchen": ["cooking_area", "galley"],
        "indoor/home/bathroom": ["restroom", "washroom"],
        "indoor/home/office": ["home_office", "study", "den"],
        "indoor/commercial": ["business", "retail", "store"],
        "indoor/commercial/office": ["workplace", "cubicle", "corporate"],
        "indoor/commercial/restaurant": ["cafe", "diner", "eatery", "bistro"],
        "indoor/commercial/gym": ["fitness_center", "health_club", "workout_room"],
        "indoor/commercial/studio": ["recording_studio", "photo_studio", "production"],
        "indoor/commercial/warehouse": ["storage", "distribution_center"],
        "indoor/public": ["public_space"],
        "indoor/public/hospital": ["medical_center", "clinic", "healthcare"],
        "indoor/public/school": ["classroom", "educational", "university", "college"],
        "indoor/public/library": ["reading_room", "study_hall"],
        "indoor/public/museum": ["gallery", "exhibition"],
        "indoor/public/theater": ["cinema", "auditorium", "concert_hall"],
        "indoor/transportation": ["vehicle_interior"],
        "indoor/transportation/car": ["automobile", "vehicle"],
        "indoor/transportation/airplane": ["aircraft", "plane_cabin"],
        "indoor/transportation/train": ["railway", "subway", "metro"],

        # Outdoor locations
        "outdoor": ["outside", "exterior", "open_air"],
        "outdoor/nature": ["natural", "wilderness", "countryside"],
        "outdoor/nature/forest": ["woods", "woodland", "trees", "jungle"],
        "outdoor/nature/beach": ["shore", "seaside", "coast", "oceanfront"],
        "outdoor/nature/mountain": ["hills", "peaks", "highlands", "alpine"],
        "outdoor/nature/desert": ["arid", "dunes", "sandy"],
        "outdoor/nature/lake": ["pond", "reservoir", "waterfront"],
        "outdoor/nature/river": ["stream", "creek", "waterway"],
        "outdoor/nature/field": ["meadow", "grassland", "prairie", "pasture"],
        "outdoor/nature/garden": ["yard", "backyard", "landscaped"],
        "outdoor/urban": ["city", "metropolitan", "downtown"],
        "outdoor/urban/street": ["road", "avenue", "boulevard", "sidewalk"],
        "outdoor/urban/park": ["plaza", "square", "public_garden"],
        "outdoor/urban/parking": ["parking_lot", "garage"],
        "outdoor/urban/construction": ["building_site", "development"],
        "outdoor/sports": ["athletic_field", "sports_venue"],
        "outdoor/sports/stadium": ["arena", "sports_complex"],
        "outdoor/sports/pool": ["swimming_pool", "aquatic_center"],
        "outdoor/sports/court": ["tennis_court", "basketball_court"],

        # Special environments
        "underwater": ["subaquatic", "marine", "diving"],
        "aerial": ["sky", "airborne", "drone_view", "birds_eye"],
        "space": ["cosmic", "astronomical", "orbit"],
    }

    # Object Tags
    OBJECT_TAGS: Dict[str, List[str]] = {
        # Electronics
        "electronics/phone": ["smartphone", "mobile", "cellphone", "iphone", "android"],
        "electronics/computer": ["laptop", "desktop", "pc", "mac", "notebook"],
        "electronics/tablet": ["ipad", "surface"],
        "electronics/camera": ["dslr", "mirrorless", "webcam", "camcorder"],
        "electronics/tv": ["television", "monitor", "screen", "display"],
        "electronics/gaming": ["console", "controller", "playstation", "xbox", "nintendo"],

        # Furniture
        "furniture/seating": ["chair", "sofa", "couch", "bench", "stool"],
        "furniture/table": ["desk", "counter", "surface"],
        "furniture/bed": ["mattress", "cot", "bunk"],
        "furniture/storage": ["cabinet", "shelf", "drawer", "closet", "wardrobe"],

        # Vehicles
        "vehicle/car": ["automobile", "sedan", "suv", "truck", "van"],
        "vehicle/motorcycle": ["motorbike", "scooter", "moped"],
        "vehicle/bicycle": ["bike", "cycle", "e-bike"],
        "vehicle/boat": ["ship", "yacht", "watercraft", "kayak"],
        "vehicle/airplane": ["aircraft", "jet", "helicopter", "drone"],

        # Food & Drink
        "food": ["meal", "dish", "cuisine"],
        "food/fruit": ["apple", "banana", "orange", "berry"],
        "food/vegetable": ["salad", "greens", "produce"],
        "food/meat": ["steak", "chicken", "beef", "pork"],
        "food/dessert": ["cake", "ice_cream", "pastry", "sweet"],
        "drink": ["beverage"],
        "drink/coffee": ["espresso", "latte", "cappuccino"],
        "drink/alcohol": ["wine", "beer", "cocktail", "spirits"],

        # Animals
        "animal/pet": ["domestic_animal"],
        "animal/pet/dog": ["puppy", "canine", "hound"],
        "animal/pet/cat": ["kitten", "feline"],
        "animal/wildlife": ["wild_animal"],
        "animal/bird": ["avian", "fowl"],
        "animal/fish": ["aquatic", "marine_life"],

        # Sports Equipment
        "sports_equipment/ball": ["football", "basketball", "soccer_ball", "tennis_ball"],
        "sports_equipment/racket": ["tennis_racket", "badminton"],
        "sports_equipment/weights": ["dumbbell", "barbell", "kettlebell"],
        "sports_equipment/mat": ["yoga_mat", "exercise_mat"],

        # Clothing & Accessories
        "clothing/casual": ["t-shirt", "jeans", "hoodie"],
        "clothing/formal": ["suit", "dress", "tie", "blazer"],
        "clothing/athletic": ["sportswear", "activewear", "jersey"],
        "accessory/jewelry": ["ring", "necklace", "bracelet", "watch"],
        "accessory/bag": ["purse", "backpack", "handbag", "luggage"],
        "accessory/glasses": ["sunglasses", "eyewear", "spectacles"],

        # Products/Commercial
        "product/cosmetics": ["makeup", "skincare", "beauty"],
        "product/package": ["box", "container", "bottle", "jar"],
        "product/brand": ["logo", "branded", "merchandise"],
    }

    # Person/People Tags
    PERSON_TAGS: Dict[str, List[str]] = {
        # Count
        "person/single": ["solo", "alone", "individual", "one_person"],
        "person/couple": ["pair", "duo", "two_people"],
        "person/group": ["crowd", "gathering", "multiple_people", "team"],

        # Demographics (inferred cautiously)
        "person/adult": ["grown_up", "mature"],
        "person/child": ["kid", "young", "minor", "youth"],
        "person/baby": ["infant", "toddler", "newborn"],

        # Roles
        "person/presenter": ["host", "anchor", "speaker", "narrator"],
        "person/athlete": ["sports_person", "player", "competitor"],
        "person/performer": ["actor", "musician", "dancer", "entertainer"],
        "person/professional": ["worker", "employee", "expert"],

        # Face characteristics
        "face/visible": ["face_shown", "frontal"],
        "face/partial": ["profile", "side_view", "obscured"],
        "face/close_up": ["portrait", "headshot"],
        "face/multiple": ["faces", "group_faces"],
    }

    # Activity Tags
    ACTIVITY_TAGS: Dict[str, List[str]] = {
        # Physical activities
        "activity/exercise": ["workout", "fitness", "training"],
        "activity/exercise/yoga": ["stretching", "meditation", "pilates"],
        "activity/exercise/running": ["jogging", "sprinting", "cardio"],
        "activity/exercise/weightlifting": ["strength_training", "gym_workout"],
        "activity/exercise/swimming": ["aquatic_exercise", "laps"],
        "activity/sports": ["athletic", "game", "competition"],
        "activity/sports/team_sport": ["soccer", "basketball", "football", "volleyball"],
        "activity/sports/individual_sport": ["tennis", "golf", "boxing", "martial_arts"],

        # Daily activities
        "activity/cooking": ["food_preparation", "baking", "culinary"],
        "activity/eating": ["dining", "meal", "snacking"],
        "activity/drinking": ["beverage_consumption"],
        "activity/cleaning": ["housework", "tidying", "organizing"],
        "activity/shopping": ["retail", "browsing", "purchasing"],

        # Work activities
        "activity/working": ["job", "employment", "professional"],
        "activity/meeting": ["conference", "discussion", "collaboration"],
        "activity/presentation": ["demo", "pitch", "lecture", "teaching"],
        "activity/typing": ["computer_work", "coding", "writing"],

        # Leisure activities
        "activity/watching": ["viewing", "spectating", "observing"],
        "activity/reading": ["studying", "browsing"],
        "activity/gaming": ["playing_games", "video_games"],
        "activity/music": ["playing_music", "listening", "concert"],
        "activity/dancing": ["movement", "choreography"],

        # Social activities
        "activity/talking": ["conversation", "discussion", "chatting"],
        "activity/celebrating": ["party", "festivity", "event"],
        "activity/traveling": ["journey", "trip", "commuting"],

        # Creative activities
        "activity/creating": ["making", "crafting", "building"],
        "activity/photography": ["shooting", "capturing", "filming"],
        "activity/art": ["painting", "drawing", "sculpting"],
    }

    # Emotion/Mood Tags
    EMOTION_TAGS: Dict[str, List[str]] = {
        "emotion/positive": ["happy", "joyful", "cheerful", "upbeat"],
        "emotion/positive/excited": ["thrilled", "enthusiastic", "energetic"],
        "emotion/positive/peaceful": ["calm", "serene", "relaxed", "tranquil"],
        "emotion/positive/loving": ["affectionate", "romantic", "tender"],
        "emotion/negative": ["sad", "unhappy", "melancholy"],
        "emotion/negative/angry": ["frustrated", "upset", "irritated"],
        "emotion/negative/fearful": ["scared", "anxious", "worried", "tense"],
        "emotion/neutral": ["calm", "composed", "balanced"],
        "emotion/dramatic": ["intense", "powerful", "emotional"],
        "emotion/humorous": ["funny", "comedic", "amusing", "entertaining"],
        "emotion/suspenseful": ["tense", "thriller", "mysterious"],
        "emotion/inspirational": ["motivating", "uplifting", "empowering"],
    }

    # Technical Quality Tags
    TECHNICAL_TAGS: Dict[str, List[str]] = {
        # Resolution
        "quality/4k": ["uhd", "2160p", "ultra_hd"],
        "quality/1080p": ["full_hd", "fhd"],
        "quality/720p": ["hd"],
        "quality/480p": ["sd", "standard_def"],
        "quality/low": ["low_res", "compressed", "pixelated"],

        # Frame rate
        "framerate/high": ["60fps", "120fps", "smooth", "high_fps"],
        "framerate/cinematic": ["24fps", "film_rate"],
        "framerate/standard": ["30fps", "normal_fps"],
        "framerate/slow_motion": ["slowmo", "high_speed"],
        "framerate/timelapse": ["time_lapse", "hyperlapse"],

        # Duration
        "duration/short": ["clip", "brief", "short_form"],
        "duration/medium": ["standard_length"],
        "duration/long": ["extended", "long_form", "feature"],

        # Image quality
        "image_quality/sharp": ["crisp", "clear", "focused"],
        "image_quality/blurry": ["soft", "out_of_focus", "motion_blur"],
        "image_quality/grainy": ["noisy", "high_iso"],
        "image_quality/overexposed": ["bright", "washed_out", "blown_out"],
        "image_quality/underexposed": ["dark", "muddy", "low_light"],

        # Stability
        "stability/stable": ["steady", "tripod", "gimbal"],
        "stability/handheld": ["shaky", "natural_movement"],
    }

    # Color Tags
    COLOR_TAGS: Dict[str, List[str]] = {
        # Primary colors
        "color/red": ["crimson", "scarlet", "ruby", "maroon"],
        "color/blue": ["azure", "navy", "cobalt", "cyan", "turquoise"],
        "color/yellow": ["gold", "amber", "lemon", "mustard"],
        "color/green": ["emerald", "olive", "lime", "teal", "mint"],
        "color/orange": ["tangerine", "coral", "peach"],
        "color/purple": ["violet", "lavender", "magenta", "plum"],
        "color/pink": ["rose", "salmon", "fuchsia"],

        # Neutral colors
        "color/white": ["ivory", "cream", "snow"],
        "color/black": ["ebony", "onyx", "jet"],
        "color/gray": ["grey", "silver", "charcoal", "slate"],
        "color/brown": ["tan", "beige", "chocolate", "bronze"],

        # Color characteristics
        "color_style/vibrant": ["saturated", "vivid", "bold", "colorful"],
        "color_style/muted": ["desaturated", "subtle", "pastel", "soft"],
        "color_style/monochrome": ["black_and_white", "grayscale", "bw"],
        "color_style/warm": ["warm_tones", "golden", "sunset"],
        "color_style/cool": ["cool_tones", "blue_cast"],
        "color_style/high_contrast": ["contrasty", "dramatic_contrast"],
        "color_style/low_contrast": ["flat", "hazy"],
    }

    # Lighting Tags
    LIGHTING_TAGS: Dict[str, List[str]] = {
        "lighting/natural": ["daylight", "sunlight", "ambient"],
        "lighting/natural/golden_hour": ["sunset", "sunrise", "warm_light"],
        "lighting/natural/blue_hour": ["twilight", "dusk", "dawn"],
        "lighting/natural/overcast": ["cloudy", "diffused", "soft_light"],
        "lighting/natural/harsh": ["midday", "direct_sun", "hard_shadows"],
        "lighting/artificial": ["studio", "indoor_lighting"],
        "lighting/artificial/fluorescent": ["office_light", "cool_white"],
        "lighting/artificial/tungsten": ["warm_indoor", "incandescent"],
        "lighting/artificial/led": ["modern_lighting"],
        "lighting/artificial/neon": ["colorful_lights", "signage"],
        "lighting/low_light": ["dim", "dark", "night", "moody"],
        "lighting/backlit": ["silhouette", "rim_light", "halo"],
        "lighting/dramatic": ["chiaroscuro", "spotlight", "contrast_lighting"],
    }

    # Composition Tags
    COMPOSITION_TAGS: Dict[str, List[str]] = {
        # Shot types
        "shot/close_up": ["tight_shot", "detail", "macro"],
        "shot/medium": ["mid_shot", "waist_up"],
        "shot/wide": ["establishing", "full_shot", "long_shot"],
        "shot/extreme_wide": ["panoramic", "landscape_shot"],
        "shot/overhead": ["top_down", "birds_eye", "aerial"],
        "shot/low_angle": ["worms_eye", "looking_up"],
        "shot/high_angle": ["looking_down", "elevated"],
        "shot/dutch_angle": ["tilted", "canted"],
        "shot/pov": ["point_of_view", "first_person", "subjective"],

        # Movement
        "camera/static": ["locked", "stationary", "fixed"],
        "camera/pan": ["horizontal_movement", "panning"],
        "camera/tilt": ["vertical_movement", "tilting"],
        "camera/zoom": ["zooming", "push_in", "pull_out"],
        "camera/tracking": ["follow", "dolly", "slider"],
        "camera/handheld": ["organic", "documentary_style"],
        "camera/drone": ["aerial_footage", "flying_camera"],

        # Framing
        "framing/centered": ["symmetrical", "balanced"],
        "framing/rule_of_thirds": ["off_center", "dynamic"],
        "framing/leading_lines": ["perspective", "depth"],
        "framing/frame_within_frame": ["nested", "layered"],
    }

    # Style/Genre Tags
    STYLE_TAGS: Dict[str, List[str]] = {
        # Video style
        "style/professional": ["polished", "high_production", "commercial"],
        "style/amateur": ["homemade", "casual", "user_generated"],
        "style/documentary": ["real", "authentic", "journalistic"],
        "style/cinematic": ["film_like", "movie_quality", "theatrical"],
        "style/vlog": ["personal", "diary", "talking_head"],
        "style/tutorial": ["how_to", "educational", "instructional"],
        "style/review": ["critique", "analysis", "opinion"],
        "style/interview": ["conversation", "q_and_a", "discussion"],
        "style/animation": ["animated", "cartoon", "motion_graphics"],
        "style/screencast": ["screen_recording", "software_demo"],

        # Genre
        "genre/entertainment": ["fun", "leisure", "amusement"],
        "genre/educational": ["learning", "informative", "knowledge"],
        "genre/news": ["current_events", "journalism", "report"],
        "genre/sports": ["athletics", "competition", "game"],
        "genre/music": ["song", "performance", "concert"],
        "genre/gaming": ["gameplay", "esports", "let_play"],
        "genre/travel": ["adventure", "exploration", "destination"],
        "genre/food": ["cooking", "recipe", "culinary"],
        "genre/beauty": ["makeup", "fashion", "style"],
        "genre/fitness": ["health", "workout", "wellness"],
        "genre/tech": ["technology", "gadget", "review"],
        "genre/comedy": ["humor", "funny", "sketch"],
        "genre/drama": ["narrative", "story", "emotional"],
        "genre/horror": ["scary", "thriller", "suspense"],
    }

    def __init__(self):
        """Initialize the tag taxonomy with all tag categories."""
        self._all_tags: Dict[str, TagDefinition] = {}
        self._aliases: Dict[str, str] = {}  # alias -> canonical tag
        self._hierarchy: Dict[str, List[str]] = {}  # parent -> children
        self._build_taxonomy()

    def _build_taxonomy(self):
        """Build the complete tag taxonomy from definitions."""
        all_categories = [
            (TagCategory.SCENE, self.SCENE_TAGS),
            (TagCategory.OBJECT, self.OBJECT_TAGS),
            (TagCategory.PERSON, self.PERSON_TAGS),
            (TagCategory.ACTIVITY, self.ACTIVITY_TAGS),
            (TagCategory.EMOTION, self.EMOTION_TAGS),
            (TagCategory.TECHNICAL, self.TECHNICAL_TAGS),
            (TagCategory.COLOR, self.COLOR_TAGS),
            (TagCategory.LIGHTING, self.LIGHTING_TAGS),
            (TagCategory.COMPOSITION, self.COMPOSITION_TAGS),
            (TagCategory.STYLE, self.STYLE_TAGS),
        ]

        for category, tag_dict in all_categories:
            for tag_name, aliases in tag_dict.items():
                # Determine parent from hierarchical name
                parts = tag_name.split("/")
                parent = "/".join(parts[:-1]) if len(parts) > 1 else None

                # Create tag definition
                tag_def = TagDefinition(
                    name=tag_name,
                    category=category,
                    aliases=aliases,
                    parent=parent,
                )
                self._all_tags[tag_name] = tag_def

                # Register aliases
                for alias in aliases:
                    self._aliases[alias.lower()] = tag_name

                # Build hierarchy
                if parent:
                    if parent not in self._hierarchy:
                        self._hierarchy[parent] = []
                    self._hierarchy[parent].append(tag_name)

    def get_tag(self, name: str) -> Optional[TagDefinition]:
        """Get a tag definition by name or alias."""
        # Check canonical name
        if name in self._all_tags:
            return self._all_tags[name]

        # Check aliases
        canonical = self._aliases.get(name.lower())
        if canonical:
            return self._all_tags[canonical]

        return None

    def get_canonical_name(self, name_or_alias: str) -> Optional[str]:
        """Get the canonical tag name from a name or alias."""
        if name_or_alias in self._all_tags:
            return name_or_alias
        return self._aliases.get(name_or_alias.lower())

    def get_children(self, parent_tag: str) -> List[str]:
        """Get all child tags of a parent tag."""
        return self._hierarchy.get(parent_tag, [])

    def get_all_descendants(self, parent_tag: str) -> Set[str]:
        """Get all descendant tags (recursive)."""
        descendants = set()
        children = self.get_children(parent_tag)
        for child in children:
            descendants.add(child)
            descendants.update(self.get_all_descendants(child))
        return descendants

    def get_ancestors(self, tag_name: str) -> List[str]:
        """Get all ancestor tags (from immediate parent to root)."""
        ancestors = []
        tag = self.get_tag(tag_name)
        while tag and tag.parent:
            ancestors.append(tag.parent)
            tag = self.get_tag(tag.parent)
        return ancestors

    def get_tags_by_category(self, category: TagCategory) -> List[str]:
        """Get all tags in a specific category."""
        return [
            name for name, tag in self._all_tags.items()
            if tag.category == category
        ]

    def search_tags(self, query: str) -> List[str]:
        """Search tags by partial name or alias match."""
        query_lower = query.lower()
        matches = []

        # Search in tag names
        for name in self._all_tags:
            if query_lower in name.lower():
                matches.append(name)

        # Search in aliases
        for alias, canonical in self._aliases.items():
            if query_lower in alias and canonical not in matches:
                matches.append(canonical)

        return matches

    def normalize_tags(self, tags: List[str]) -> List[str]:
        """Normalize a list of tags to their canonical forms."""
        normalized = []
        for tag in tags:
            canonical = self.get_canonical_name(tag)
            if canonical and canonical not in normalized:
                normalized.append(canonical)
        return normalized

    def expand_tags_with_ancestors(self, tags: List[str]) -> List[str]:
        """Expand tags to include all ancestor tags."""
        expanded = set(tags)
        for tag in tags:
            expanded.update(self.get_ancestors(tag))
        return list(expanded)

    def get_tag_statistics(self) -> Dict[str, Any]:
        """Get statistics about the tag taxonomy."""
        category_counts = {}
        for tag in self._all_tags.values():
            cat_name = tag.category.value
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1

        return {
            "total_tags": len(self._all_tags),
            "total_aliases": len(self._aliases),
            "categories": category_counts,
            "hierarchy_depth": self._calculate_max_depth(),
        }

    def _calculate_max_depth(self) -> int:
        """Calculate the maximum hierarchy depth."""
        max_depth = 0
        for tag_name in self._all_tags:
            depth = len(tag_name.split("/"))
            max_depth = max(max_depth, depth)
        return max_depth


# Global taxonomy instance
_taxonomy: Optional[TagTaxonomy] = None


def get_taxonomy() -> TagTaxonomy:
    """Get the global tag taxonomy instance."""
    global _taxonomy
    if _taxonomy is None:
        _taxonomy = TagTaxonomy()
    return _taxonomy
