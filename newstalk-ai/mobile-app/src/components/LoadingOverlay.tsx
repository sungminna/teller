import React from 'react';
import { View, StyleSheet } from 'react-native';
import { ActivityIndicator, Text, useTheme } from 'react-native-paper';

interface LoadingOverlayProps {
  message?: string;
  visible?: boolean;
}

export default function LoadingOverlay({
  message = 'Loading...',
  visible = true,
}: LoadingOverlayProps) {
  const theme = useTheme();

  if (!visible) return null;

  return (
    <View style={[styles.container, { backgroundColor: theme.colors.background }]}>
      <ActivityIndicator size="large" color={theme.colors.primary} />
      <Text variant="bodyLarge" style={[styles.message, { color: theme.colors.onBackground }]}>
        {message}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  message: {
    marginTop: 16,
    textAlign: 'center',
  },
}); 