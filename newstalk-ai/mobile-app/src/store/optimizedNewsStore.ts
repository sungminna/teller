/**
 * 최적화된 뉴스 상태 관리 스토어
 * - 정규화된 데이터 구조로 메모리 효율성 향상
 * - 메모이제이션된 셀렉터로 불필요한 리렌더링 방지
 * - 효율적인 업데이트 로직으로 성능 최적화
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { devtools } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { useCallback } from 'react';

// 타입 정의
export interface NewsItem {
  id: string;
  title: string;
  summary: string;
  content?: string;
  category: string;
  source: string;
  publishedAt: string;
  trendingScore: number;
  audioUrl?: string;
  duration?: number;
  quality: number;
  isBookmarked?: boolean;
  isRead?: boolean;
  readProgress?: number;
}

export interface NewsState {
  // 정규화된 데이터 구조 - 메모리 효율성
  entities: Record<string, NewsItem>;
  ids: string[];
  
  // 카테고리별 ID 목록으로 중복 방지
  categorizedIds: Record<string, string[]>;
  
  // 세분화된 로딩 상태
  loading: {
    trending: boolean;
    personalized: boolean;
    category: Record<string, boolean>;
    search: boolean;
  };
  
  // 에러 상태 관리
  errors: {
    trending?: string;
    personalized?: string;
    category: Record<string, string>;
    search?: string;
  };
  
  // 캐시 메타데이터
  metadata: {
    lastFetch: Record<string, number>;
    hasMore: Record<string, boolean>;
    totalCount: Record<string, number>;
  };
  
  // 검색 관련 상태
  searchQuery: string;
  searchResults: string[];
  
  // 필터 및 정렬 상태
  activeCategory: string | null;
  sortBy: 'trending' | 'publishedAt' | 'quality';
}

// 액션 인터페이스
export interface NewsActions {
  setNews: (news: NewsItem[], type: 'trending' | 'personalized' | 'category', category?: string) => void;
  addNews: (news: NewsItem[]) => void;
  updateNewsItem: (id: string, updates: Partial<NewsItem>) => void;
  removeNewsItem: (id: string) => void;
  setLoading: (type: keyof NewsState['loading'] | string, isLoading: boolean) => void;
  setError: (type: keyof NewsState['errors'] | string, error: string | null) => void;
  clearErrors: () => void;
  setSearchQuery: (query: string) => void;
  setSearchResults: (results: string[]) => void;
  clearSearch: () => void;
  setActiveCategory: (category: string | null) => void;
  setSortBy: (sortBy: NewsState['sortBy']) => void;
  toggleBookmark: (id: string) => void;
  markAsRead: (id: string, progress?: number) => void;
  refreshNews: (type: 'trending' | 'personalized' | 'category', category?: string) => Promise<void>;
  clearCache: () => void;
  optimizeStore: () => void;
}

// 초기 상태
const initialState: NewsState = {
  entities: {},
  ids: [],
  categorizedIds: {},
  loading: {
    trending: false,
    personalized: false,
    category: {},
    search: false,
  },
  errors: {
    category: {},
  },
  metadata: {
    lastFetch: {},
    hasMore: {},
    totalCount: {},
  },
  searchQuery: '',
  searchResults: [],
  activeCategory: null,
  sortBy: 'trending',
};

// 최적화된 뉴스 스토어
export const useOptimizedNewsStore = create<NewsState & NewsActions>()(
  subscribeWithSelector(
    devtools(
      immer((set, get) => ({
        ...initialState,
        
        setNews: (news: NewsItem[], type: 'trending' | 'personalized' | 'category', category?: string) => 
          set((state) => {
            const entities = { ...state.entities };
            const newIds: string[] = [];
            
            // 정규화 처리
            news.forEach(item => {
              entities[item.id] = item;
              newIds.push(item.id);
            });
            
            state.entities = entities;
            
            if (type === 'category' && category) {
              state.categorizedIds[category] = newIds;
            } else if (type === 'trending') {
              state.ids = newIds;
            }
            
            // 메타데이터 업데이트
            const key = type === 'category' ? `${type}_${category}` : type;
            state.metadata.lastFetch[key] = Date.now();
            state.metadata.totalCount[key] = news.length;
            
            // 로딩 상태 해제
            if (type === 'category' && category) {
              state.loading.category[category] = false;
            } else {
              state.loading[type] = false;
            }
          }),
        
        addNews: (news: NewsItem[]) => 
          set((state) => {
            news.forEach(item => {
              if (!state.entities[item.id]) {
                state.entities[item.id] = item;
                state.ids.push(item.id);
              }
            });
          }),
        
        updateNewsItem: (id: string, updates: Partial<NewsItem>) => 
          set((state) => {
            if (state.entities[id]) {
              Object.assign(state.entities[id], updates);
            }
          }),
        
        removeNewsItem: (id: string) => 
          set((state) => {
            delete state.entities[id];
            state.ids = state.ids.filter(itemId => itemId !== id);
            
            Object.keys(state.categorizedIds).forEach(category => {
              state.categorizedIds[category] = state.categorizedIds[category].filter(
                itemId => itemId !== id
              );
            });
            
            state.searchResults = state.searchResults.filter(itemId => itemId !== id);
          }),
        
        setLoading: (type: keyof NewsState['loading'] | string, isLoading: boolean) => 
          set((state) => {
            if (type in state.loading) {
              (state.loading as any)[type] = isLoading;
            } else {
              state.loading.category[type] = isLoading;
            }
          }),
        
        setError: (type: keyof NewsState['errors'] | string, error: string | null) => 
          set((state) => {
            if (type in state.errors) {
              (state.errors as any)[type] = error;
            } else {
              if (error) {
                state.errors.category[type] = error;
              } else {
                delete state.errors.category[type];
              }
            }
          }),
        
        clearErrors: () => 
          set((state) => {
            state.errors = { category: {} };
          }),
        
        setSearchQuery: (query: string) => 
          set((state) => {
            state.searchQuery = query;
          }),
        
        setSearchResults: (results: string[]) => 
          set((state) => {
            state.searchResults = results;
          }),
        
        clearSearch: () => 
          set((state) => {
            state.searchQuery = '';
            state.searchResults = [];
            state.loading.search = false;
            state.errors.search = undefined;
          }),
        
        setActiveCategory: (category: string | null) => 
          set((state) => {
            state.activeCategory = category;
          }),
        
        setSortBy: (sortBy: NewsState['sortBy']) => 
          set((state) => {
            state.sortBy = sortBy;
          }),
        
        toggleBookmark: (id: string) => 
          set((state) => {
            if (state.entities[id]) {
              state.entities[id].isBookmarked = !state.entities[id].isBookmarked;
            }
          }),
        
        markAsRead: (id: string, progress = 100) => 
          set((state) => {
            if (state.entities[id]) {
              state.entities[id].isRead = progress >= 100;
              state.entities[id].readProgress = progress;
            }
          }),
        
        refreshNews: async (type: 'trending' | 'personalized' | 'category', category?: string) => {
          const { setLoading, setError } = get();
          
          try {
            const key = category ? category : type;
            setLoading(key, true);
            
            // API 호출 로직은 실제 구현에서 추가
            // const news = await fetchNews(type, category);
            // get().setNews(news, type, category);
            
          } catch (error) {
            const key = category ? category : type;
            setError(key, error instanceof Error ? error.message : 'Unknown error');
          }
        },
        
        clearCache: () => 
          set(() => ({ ...initialState })),
        
        optimizeStore: () => 
          set((state) => {
            // 1시간 이상 된 메타데이터 정리
            const now = Date.now();
            const oneHour = 60 * 60 * 1000;
            
            Object.keys(state.metadata.lastFetch).forEach(key => {
              if (now - state.metadata.lastFetch[key] > oneHour) {
                delete state.metadata.lastFetch[key];
                delete state.metadata.hasMore[key];
                delete state.metadata.totalCount[key];
              }
            });
            
            // 읽은 뉴스 중 오래된 것들 제거 (100개 초과 시)
            const readItems = Object.keys(state.entities).filter(
              id => state.entities[id].isRead
            );
            
            if (readItems.length > 100) {
              const sortedReadItems = readItems
                .sort((a, b) => 
                  new Date(state.entities[a].publishedAt).getTime() - 
                  new Date(state.entities[b].publishedAt).getTime()
                )
                .slice(0, 50);
              
              sortedReadItems.forEach(id => {
                delete state.entities[id];
                state.ids = state.ids.filter(itemId => itemId !== id);
              });
            }
          }),
      })),
      {
        name: 'optimized-news-store',
        partialize: (state) => ({
          entities: state.entities,
          ids: state.ids,
          categorizedIds: state.categorizedIds,
          metadata: state.metadata,
        }),
      }
    )
  )
);

// 메모이제이션된 셀렉터
export const useNewsSelectors = () => {
  const selectNewsByCategory = useCallback(
    (category: string) => 
      useOptimizedNewsStore(
        (state) => {
          const ids = state.categorizedIds[category] || [];
          return ids.map(id => state.entities[id]).filter(Boolean);
        }
      ),
    []
  );
  
  const selectSortedNews = useCallback(
    () => 
      useOptimizedNewsStore(
        (state) => {
          const news = state.ids.map(id => state.entities[id]).filter(Boolean);
          
          switch (state.sortBy) {
            case 'trending':
              return news.sort((a, b) => b.trendingScore - a.trendingScore);
            case 'publishedAt':
              return news.sort((a, b) => 
                new Date(b.publishedAt).getTime() - new Date(a.publishedAt).getTime()
              );
            case 'quality':
              return news.sort((a, b) => b.quality - a.quality);
            default:
              return news;
          }
        }
      ),
    []
  );
  
  const selectSearchResults = useCallback(
    () => 
      useOptimizedNewsStore(
        (state) => 
          state.searchResults.map(id => state.entities[id]).filter(Boolean)
      ),
    []
  );
  
  const selectBookmarkedNews = useCallback(
    () => 
      useOptimizedNewsStore(
        (state) => 
          Object.values(state.entities).filter(news => news.isBookmarked)
      ),
    []
  );
  
  return {
    selectNewsByCategory,
    selectSortedNews,
    selectSearchResults,
    selectBookmarkedNews,
  };
}; 