import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Text, Button, useTheme } from 'react-native-paper';
import { Ionicons } from '@expo/vector-icons';

interface EmptyStateProps {
  title: string;
  description?: string;
  icon?: keyof typeof Ionicons.glyphMap;
  action?: {
    text: string;
    onPress: () => void;
  };
}

export default function EmptyState({
  title,
  description,
  icon = 'newspaper-outline',
  action,
}: EmptyStateProps) {
  const theme = useTheme();

  return (
    <View style={styles.container}>
      <Ionicons
        name={icon}
        size={64}
        color={theme.colors.onSurfaceVariant}
        style={styles.icon}
      />
      <Text
        variant="headlineSmall"
        style={[styles.title, { color: theme.colors.onSurface }]}
      >
        {title}
      </Text>
      {description && (
        <Text
          variant="bodyMedium"
          style={[styles.description, { color: theme.colors.onSurfaceVariant }]}
        >
          {description}
        </Text>
      )}
      {action && (
        <Button
          mode="contained"
          onPress={action.onPress}
          style={styles.action}
        >
          {action.text}
        </Button>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 32,
  },
  icon: {
    marginBottom: 16,
    opacity: 0.6,
  },
  title: {
    marginBottom: 8,
    textAlign: 'center',
  },
  description: {
    marginBottom: 24,
    textAlign: 'center',
    lineHeight: 20,
  },
  action: {
    marginTop: 8,
  },
}); 