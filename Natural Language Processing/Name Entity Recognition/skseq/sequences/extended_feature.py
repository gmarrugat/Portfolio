
from skseq.sequences.id_feature import IDFeatures
from skseq.sequences.id_feature import UnicodeFeatures
from util import hasdigits, is_short_word, is_medium_short_word, is_medium_long_word, is_long_word

# ----------
# Feature Class
# Extracts features from a labeled corpus (only supported features are extracted
# ----------

class ExtendedFeatures(IDFeatures):

    def add_emission_features(self, sequence, pos, y, features):
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)

        # Get word name from ID.
        if isinstance(x, str):
            x_name = x
        else:
            x_name = self.dataset.x_dict.get_label_name(x)

        word = str(x_name)
        # Generate feature name.
        feat_name = "id:%s::%s" % (word, y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        # word is a number
        if word.isdigit():
            feat_name = "isdigit::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # word has digits
        if hasdigits(word):
            feat_name = "hasdigits::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # first character is capital letter
        if word.istitle():
            feat_name = "istitle::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # word isupper
        if word.isupper():
            feat_name = "allupper::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # word has hyphen
        if word.find("-") != -1:
            # Generate feature name.
            feat_name = "hyphen::%s" % y_name
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        # word has apostrophe
        if word.find("'") != -1:
            feat_name = "apostrophe::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # word has dots
        if word.find(".") != -1:
            feat_name = "dots::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # word has less than 4 characters
        if is_short_word(word):
            feat_name = "is_short_word::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # word has between 3 and 5 characters
        if is_medium_short_word(word):
            feat_name = "is_medium_short_word::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # word has between 6 and 7 characters
        if is_medium_long_word(word):
            feat_name = "is_medium_long_word::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        # word has more than 7 characters
        if is_long_word(word):
            feat_name = "is_long_word::%s" % y_name
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        return features
