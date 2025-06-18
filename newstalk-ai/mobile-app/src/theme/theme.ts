import { MD3LightTheme, MD3DarkTheme } from 'react-native-paper';

// Custom color palette
const colors = {
  primary: '#1976D2',
  primaryContainer: '#E3F2FD',
  secondary: '#03DAC6',
  secondaryContainer: '#E0F2F1',
  tertiary: '#FF9800',
  tertiaryContainer: '#FFF3E0',
  surface: '#FFFFFF',
  surfaceVariant: '#F5F5F5',
  background: '#FAFAFA',
  error: '#F44336',
  errorContainer: '#FFEBEE',
  onPrimary: '#FFFFFF',
  onPrimaryContainer: '#0D47A1',
  onSecondary: '#000000',
  onSecondaryContainer: '#004D40',
  onTertiary: '#000000',
  onTertiaryContainer: '#E65100',
  onSurface: '#1C1B1F',
  onSurfaceVariant: '#49454F',
  onError: '#FFFFFF',
  onErrorContainer: '#410E0B',
  onBackground: '#1C1B1F',
  outline: '#79747E',
  outlineVariant: '#CAC4D0',
  inverseSurface: '#313033',
  inverseOnSurface: '#F4EFF4',
  inversePrimary: '#90CAF9',
  shadow: '#000000',
  scrim: '#000000',
  surfaceDisabled: 'rgba(28, 27, 31, 0.12)',
  onSurfaceDisabled: 'rgba(28, 27, 31, 0.38)',
  backdrop: 'rgba(73, 69, 79, 0.4)',
};

const darkColors = {
  primary: '#90CAF9',
  primaryContainer: '#0D47A1',
  secondary: '#4DB6AC',
  secondaryContainer: '#004D40',
  tertiary: '#FFB74D',
  tertiaryContainer: '#E65100',
  surface: '#121212',
  surfaceVariant: '#1E1E1E',
  background: '#0A0A0A',
  error: '#FF5252',
  errorContainer: '#410E0B',
  onPrimary: '#0D47A1',
  onPrimaryContainer: '#E3F2FD',
  onSecondary: '#004D40',
  onSecondaryContainer: '#E0F2F1',
  onTertiary: '#E65100',
  onTertiaryContainer: '#FFF3E0',
  onSurface: '#E6E1E5',
  onSurfaceVariant: '#C4C7C5',
  onError: '#410E0B',
  onErrorContainer: '#FFEBEE',
  onBackground: '#E6E1E5',
  outline: '#938F99',
  outlineVariant: '#49454F',
  inverseSurface: '#E6E1E5',
  inverseOnSurface: '#313033',
  inversePrimary: '#1976D2',
  shadow: '#000000',
  scrim: '#000000',
  surfaceDisabled: 'rgba(230, 225, 229, 0.12)',
  onSurfaceDisabled: 'rgba(230, 225, 229, 0.38)',
  backdrop: 'rgba(73, 69, 79, 0.4)',
};

export const lightTheme = {
  ...MD3LightTheme,
  colors: {
    ...MD3LightTheme.colors,
    ...colors,
  },
  roundness: 12,
  fonts: {
    ...MD3LightTheme.fonts,
    displayLarge: {
      ...MD3LightTheme.fonts.displayLarge,
      fontFamily: 'System',
    },
    displayMedium: {
      ...MD3LightTheme.fonts.displayMedium,
      fontFamily: 'System',
    },
    displaySmall: {
      ...MD3LightTheme.fonts.displaySmall,
      fontFamily: 'System',
    },
    headlineLarge: {
      ...MD3LightTheme.fonts.headlineLarge,
      fontFamily: 'System',
    },
    headlineMedium: {
      ...MD3LightTheme.fonts.headlineMedium,
      fontFamily: 'System',
    },
    headlineSmall: {
      ...MD3LightTheme.fonts.headlineSmall,
      fontFamily: 'System',
    },
    titleLarge: {
      ...MD3LightTheme.fonts.titleLarge,
      fontFamily: 'System',
    },
    titleMedium: {
      ...MD3LightTheme.fonts.titleMedium,
      fontFamily: 'System',
    },
    titleSmall: {
      ...MD3LightTheme.fonts.titleSmall,
      fontFamily: 'System',
    },
    labelLarge: {
      ...MD3LightTheme.fonts.labelLarge,
      fontFamily: 'System',
    },
    labelMedium: {
      ...MD3LightTheme.fonts.labelMedium,
      fontFamily: 'System',
    },
    labelSmall: {
      ...MD3LightTheme.fonts.labelSmall,
      fontFamily: 'System',
    },
    bodyLarge: {
      ...MD3LightTheme.fonts.bodyLarge,
      fontFamily: 'System',
    },
    bodyMedium: {
      ...MD3LightTheme.fonts.bodyMedium,
      fontFamily: 'System',
    },
    bodySmall: {
      ...MD3LightTheme.fonts.bodySmall,
      fontFamily: 'System',
    },
  },
};

export const darkTheme = {
  ...MD3DarkTheme,
  colors: {
    ...MD3DarkTheme.colors,
    ...darkColors,
  },
  roundness: 12,
  fonts: {
    ...MD3DarkTheme.fonts,
    displayLarge: {
      ...MD3DarkTheme.fonts.displayLarge,
      fontFamily: 'System',
    },
    displayMedium: {
      ...MD3DarkTheme.fonts.displayMedium,
      fontFamily: 'System',
    },
    displaySmall: {
      ...MD3DarkTheme.fonts.displaySmall,
      fontFamily: 'System',
    },
    headlineLarge: {
      ...MD3DarkTheme.fonts.headlineLarge,
      fontFamily: 'System',
    },
    headlineMedium: {
      ...MD3DarkTheme.fonts.headlineMedium,
      fontFamily: 'System',
    },
    headlineSmall: {
      ...MD3DarkTheme.fonts.headlineSmall,
      fontFamily: 'System',
    },
    titleLarge: {
      ...MD3DarkTheme.fonts.titleLarge,
      fontFamily: 'System',
    },
    titleMedium: {
      ...MD3DarkTheme.fonts.titleMedium,
      fontFamily: 'System',
    },
    titleSmall: {
      ...MD3DarkTheme.fonts.titleSmall,
      fontFamily: 'System',
    },
    labelLarge: {
      ...MD3DarkTheme.fonts.labelLarge,
      fontFamily: 'System',
    },
    labelMedium: {
      ...MD3DarkTheme.fonts.labelMedium,
      fontFamily: 'System',
    },
    labelSmall: {
      ...MD3DarkTheme.fonts.labelSmall,
      fontFamily: 'System',
    },
    bodyLarge: {
      ...MD3DarkTheme.fonts.bodyLarge,
      fontFamily: 'System',
    },
    bodyMedium: {
      ...MD3DarkTheme.fonts.bodyMedium,
      fontFamily: 'System',
    },
    bodySmall: {
      ...MD3DarkTheme.fonts.bodySmall,
      fontFamily: 'System',
    },
  },
};

export type AppTheme = typeof lightTheme; 