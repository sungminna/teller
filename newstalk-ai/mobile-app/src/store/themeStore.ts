import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Appearance } from 'react-native';

export type ThemeMode = 'light' | 'dark' | 'system';

interface ThemeState {
  theme: 'light' | 'dark';
  mode: ThemeMode;
  setTheme: (theme: 'light' | 'dark') => void;
  setMode: (mode: ThemeMode) => void;
  initializeTheme: () => Promise<void>;
}

export const useThemeStore = create<ThemeState>()(
  persist(
    (set, get) => ({
      theme: 'light',
      mode: 'system',
      
      setTheme: (theme) => {
        set({ theme });
      },
      
      setMode: (mode) => {
        set({ mode });
        
        if (mode === 'system') {
          const systemTheme = Appearance.getColorScheme() || 'light';
          set({ theme: systemTheme });
        }
      },
      
      initializeTheme: async () => {
        const { mode } = get();
        
        if (mode === 'system') {
          const systemTheme = Appearance.getColorScheme() || 'light';
          set({ theme: systemTheme });
          
          // Listen for system theme changes
          const subscription = Appearance.addChangeListener(({ colorScheme }) => {
            if (get().mode === 'system') {
              set({ theme: colorScheme || 'light' });
            }
          });
          
          return () => subscription.remove();
        }
      },
    }),
    {
      name: 'theme-storage',
      storage: createJSONStorage(() => AsyncStorage),
    }
  )
); 