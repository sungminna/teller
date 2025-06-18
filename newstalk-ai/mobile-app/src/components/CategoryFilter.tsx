import React from 'react';
import { View, StyleSheet, ScrollView } from 'react-native';
import { Chip, useTheme } from 'react-native-paper';

import { NewsCategory } from '../store/newsStore';

interface CategoryFilterProps {
  categories: NewsCategory[];
  selectedCategory: string | null;
  onCategorySelect: (categoryId: string) => void;
  style?: any;
}

export default function CategoryFilter({
  categories,
  selectedCategory,
  onCategorySelect,
  style,
}: CategoryFilterProps) {
  const theme = useTheme();

  return (
    <View style={[styles.container, style]}>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {categories.map((category) => (
          <Chip
            key={category.id}
            selected={selectedCategory === category.id || (!selectedCategory && category.id === 'all')}
            onPress={() => onCategorySelect(category.id)}
            style={styles.chip}
            mode="outlined"
          >
            {category.name}
          </Chip>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingVertical: 8,
  },
  scrollContent: {
    paddingHorizontal: 16,
    gap: 8,
  },
  chip: {
    marginRight: 8,
  },
}); 