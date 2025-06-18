import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as SecureStore from 'expo-secure-store';

export interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  preferences: {
    interests: string[];
    voiceStyle: string;
    playbackSpeed: number;
    notificationsEnabled: boolean;
    offlineMode: boolean;
  };
}

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  accessToken: string | null;
  refreshToken: string | null;
  isLoading: boolean;
  
  // Actions
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  refreshAccessToken: () => Promise<void>;
  updateUser: (userData: Partial<User>) => void;
  initializeAuth: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      isAuthenticated: false,
      accessToken: null,
      refreshToken: null,
      isLoading: false,
      
      login: async (email: string, password: string) => {
        set({ isLoading: true });
        
        try {
          const response = await fetch(`${process.env.EXPO_PUBLIC_API_URL}/api/users/login`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password }),
          });
          
          if (!response.ok) {
            throw new Error('Login failed');
          }
          
          const data = await response.json();
          
          // Store tokens securely
          await SecureStore.setItemAsync('accessToken', data.access_token);
          await SecureStore.setItemAsync('refreshToken', data.refresh_token);
          
          set({
            user: data.user,
            isAuthenticated: true,
            accessToken: data.access_token,
            refreshToken: data.refresh_token,
            isLoading: false,
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },
      
      logout: async () => {
        try {
          // Clear secure storage
          await SecureStore.deleteItemAsync('accessToken');
          await SecureStore.deleteItemAsync('refreshToken');
          
          set({
            user: null,
            isAuthenticated: false,
            accessToken: null,
            refreshToken: null,
          });
        } catch (error) {
          console.error('Logout error:', error);
        }
      },
      
      register: async (email: string, password: string, name: string) => {
        set({ isLoading: true });
        
        try {
          const response = await fetch(`${process.env.EXPO_PUBLIC_API_URL}/api/users/register`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password, name }),
          });
          
          if (!response.ok) {
            throw new Error('Registration failed');
          }
          
          const data = await response.json();
          
          // Store tokens securely
          await SecureStore.setItemAsync('accessToken', data.access_token);
          await SecureStore.setItemAsync('refreshToken', data.refresh_token);
          
          set({
            user: data.user,
            isAuthenticated: true,
            accessToken: data.access_token,
            refreshToken: data.refresh_token,
            isLoading: false,
          });
        } catch (error) {
          set({ isLoading: false });
          throw error;
        }
      },
      
      refreshAccessToken: async () => {
        const { refreshToken } = get();
        
        if (!refreshToken) {
          throw new Error('No refresh token available');
        }
        
        try {
          const response = await fetch(`${process.env.EXPO_PUBLIC_API_URL}/api/users/refresh`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${refreshToken}`,
            },
          });
          
          if (!response.ok) {
            throw new Error('Token refresh failed');
          }
          
          const data = await response.json();
          
          // Update tokens
          await SecureStore.setItemAsync('accessToken', data.access_token);
          
          set({
            accessToken: data.access_token,
          });
        } catch (error) {
          // If refresh fails, logout user
          get().logout();
          throw error;
        }
      },
      
      updateUser: (userData: Partial<User>) => {
        const { user } = get();
        if (user) {
          set({
            user: { ...user, ...userData },
          });
        }
      },
      
      initializeAuth: async () => {
        try {
          // Get tokens from secure storage
          const accessToken = await SecureStore.getItemAsync('accessToken');
          const refreshToken = await SecureStore.getItemAsync('refreshToken');
          
          if (accessToken && refreshToken) {
            // Verify token validity
            const response = await fetch(`${process.env.EXPO_PUBLIC_API_URL}/api/users/me`, {
              headers: {
                'Authorization': `Bearer ${accessToken}`,
              },
            });
            
            if (response.ok) {
              const userData = await response.json();
              set({
                user: userData,
                isAuthenticated: true,
                accessToken,
                refreshToken,
              });
            } else {
              // Try to refresh token
              await get().refreshAccessToken();
            }
          }
        } catch (error) {
          console.error('Auth initialization error:', error);
          // Clear invalid tokens
          await get().logout();
        }
      },
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => AsyncStorage),
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
); 