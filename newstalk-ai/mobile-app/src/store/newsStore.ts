import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';

export interface NewsItem {
  id: string;
  title: string;
  summary: string;
  category: string;
  audioUrl?: string;
  duration?: number;
  publishedAt: string;
  isBookmarked: boolean;
  isDownloaded: boolean;
  processingStatus: 'pending' | 'processing' | 'completed' | 'failed';
  quality: number;
  personalizedScore: number;
}

export interface NewsCategory {
  id: string;
  name: string;
  isSelected: boolean;
}

interface NewsState {
  // Data
  news: NewsItem[];
  categories: NewsCategory[];
  currentNews: NewsItem | null;
  
  // UI State
  isLoading: boolean;
  isRefreshing: boolean;
  selectedCategory: string | null;
  searchQuery: string;
  
  // Processing
  processingThreads: Map<string, string>; // newsId -> threadId
  
  // Offline
  offlineNews: NewsItem[];
  lastSyncTime: number;
  
  // Actions
  fetchNews: (category?: string) => Promise<void>;
  refreshNews: () => Promise<void>;
  processNews: (newsId: string) => Promise<void>;
  toggleBookmark: (newsId: string) => void;
  downloadForOffline: (newsId: string) => Promise<void>;
  setCurrentNews: (news: NewsItem | null) => void;
  setSelectedCategory: (category: string | null) => void;
  setSearchQuery: (query: string) => void;
  updateCategories: (categories: NewsCategory[]) => void;
  submitFeedback: (newsId: string, rating: number, comment?: string) => Promise<void>;
  
  // SSE Connection
  connectToUpdates: (userId: string) => void;
  disconnectFromUpdates: () => void;
}

export const useNewsStore = create<NewsState>()(
  persist(
    (set, get) => ({
      // Initial state
      news: [],
      categories: [
        { id: 'all', name: '전체', isSelected: true },
        { id: 'politics', name: '정치', isSelected: false },
        { id: 'economy', name: '경제', isSelected: false },
        { id: 'society', name: '사회', isSelected: false },
        { id: 'culture', name: '문화', isSelected: false },
        { id: 'sports', name: '스포츠', isSelected: false },
        { id: 'tech', name: '기술', isSelected: false },
      ],
      currentNews: null,
      isLoading: false,
      isRefreshing: false,
      selectedCategory: null,
      searchQuery: '',
      processingThreads: new Map(),
      offlineNews: [],
      lastSyncTime: 0,
      
      fetchNews: async (category) => {
        set({ isLoading: true });
        
        try {
          const params = new URLSearchParams();
          if (category && category !== 'all') {
            params.append('category', category);
          }
          
          const response = await fetch(
            `${process.env.EXPO_PUBLIC_API_URL}/api/news?${params}`,
            {
              headers: {
                'Authorization': `Bearer ${get().accessToken}`,
              },
            }
          );
          
          if (!response.ok) {
            throw new Error('Failed to fetch news');
          }
          
          const newsData = await response.json();
          
          set({
            news: newsData,
            isLoading: false,
            lastSyncTime: Date.now(),
          });
        } catch (error) {
          set({ isLoading: false });
          console.error('Fetch news error:', error);
          throw error;
        }
      },
      
      refreshNews: async () => {
        set({ isRefreshing: true });
        
        try {
          await get().fetchNews(get().selectedCategory || undefined);
        } finally {
          set({ isRefreshing: false });
        }
      },
      
      processNews: async (newsId: string) => {
        try {
          const response = await fetch(
            `${process.env.EXPO_PUBLIC_API_URL}/api/news/process`,
            {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${get().accessToken}`,
              },
              body: JSON.stringify({ news_id: newsId }),
            }
          );
          
          if (!response.ok) {
            throw new Error('Failed to start news processing');
          }
          
          const { thread_id } = await response.json();
          
          // Update processing status
          set((state) => ({
            news: state.news.map((item) =>
              item.id === newsId
                ? { ...item, processingStatus: 'processing' }
                : item
            ),
            processingThreads: new Map(state.processingThreads).set(newsId, thread_id),
          }));
        } catch (error) {
          // Update status to failed
          set((state) => ({
            news: state.news.map((item) =>
              item.id === newsId
                ? { ...item, processingStatus: 'failed' }
                : item
            ),
          }));
          
          console.error('Process news error:', error);
          throw error;
        }
      },
      
      toggleBookmark: (newsId: string) => {
        set((state) => ({
          news: state.news.map((item) =>
            item.id === newsId
              ? { ...item, isBookmarked: !item.isBookmarked }
              : item
          ),
        }));
      },
      
      downloadForOffline: async (newsId: string) => {
        const newsItem = get().news.find((item) => item.id === newsId);
        if (!newsItem || !newsItem.audioUrl) return;
        
        try {
          // Download audio file
          // Implementation would depend on your file storage solution
          
          set((state) => ({
            news: state.news.map((item) =>
              item.id === newsId
                ? { ...item, isDownloaded: true }
                : item
            ),
            offlineNews: [...state.offlineNews, newsItem],
          }));
        } catch (error) {
          console.error('Download error:', error);
          throw error;
        }
      },
      
      setCurrentNews: (news) => {
        set({ currentNews: news });
      },
      
      setSelectedCategory: (category) => {
        set({ selectedCategory: category });
        get().fetchNews(category || undefined);
      },
      
      setSearchQuery: (query) => {
        set({ searchQuery: query });
      },
      
      updateCategories: (categories) => {
        set({ categories });
      },
      
      submitFeedback: async (newsId: string, rating: number, comment?: string) => {
        try {
          const response = await fetch(
            `${process.env.EXPO_PUBLIC_API_URL}/api/news/feedback`,
            {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${get().accessToken}`,
              },
              body: JSON.stringify({
                news_id: newsId,
                rating,
                comment,
              }),
            }
          );
          
          if (!response.ok) {
            throw new Error('Failed to submit feedback');
          }
        } catch (error) {
          console.error('Feedback error:', error);
          throw error;
        }
      },
      
      connectToUpdates: (userId: string) => {
        // SSE connection implementation
        const eventSource = new EventSource(
          `${process.env.EXPO_PUBLIC_API_URL}/api/news/stream/${userId}`
        );
        
        eventSource.onmessage = (event) => {
          const data = JSON.parse(event.data);
          
          if (data.type === 'news_processed') {
            set((state) => ({
              news: state.news.map((item) =>
                item.id === data.news_id
                  ? {
                      ...item,
                      processingStatus: 'completed',
                      audioUrl: data.audio_url,
                      duration: data.duration,
                      quality: data.quality,
                    }
                  : item
              ),
            }));
          }
        };
        
        eventSource.onerror = (error) => {
          console.error('SSE error:', error);
          eventSource.close();
        };
        
        // Store reference for cleanup
        (get() as any).eventSource = eventSource;
      },
      
      disconnectFromUpdates: () => {
        const eventSource = (get() as any).eventSource;
        if (eventSource) {
          eventSource.close();
        }
      },
    }),
    {
      name: 'news-storage',
      storage: createJSONStorage(() => AsyncStorage),
      partialize: (state) => ({
        categories: state.categories,
        offlineNews: state.offlineNews,
        lastSyncTime: state.lastSyncTime,
      }),
    }
  )
); 