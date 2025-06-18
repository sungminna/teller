import React, { useEffect, useState, useCallback, useMemo } from 'react';
import {
  View,
  StyleSheet,
  RefreshControl,
  Alert,
  Dimensions,
  ScrollView,
} from 'react-native';
import {
  Searchbar,
  FAB,
  Portal,
  Modal,
  Button,
  Text,
  Chip,
  ActivityIndicator,
  useTheme,
  Card,
} from 'react-native-paper';
import { SafeAreaView } from 'react-native-safe-area-context';
import { FlatGrid } from 'react-native-super-grid';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';

// Store
import { useNewsStore } from '../store/newsStore';
import { useAuthStore } from '../store/authStore';

// Components
import NewsCard from '../components/NewsCard';
import CategoryFilter from '../components/CategoryFilter';
import LoadingOverlay from '../components/LoadingOverlay';
import EmptyState from '../components/EmptyState';

// Types
import { RootStackParamList } from '../../App';
import { NewsItem } from '../store/newsStore';

type HomeScreenNavigationProp = NativeStackNavigationProp<RootStackParamList>;

const { width } = Dimensions.get('window');

export default function HomeScreen() {
  const theme = useTheme();
  const navigation = useNavigation<HomeScreenNavigationProp>();
  
  // Store state
  const {
    news,
    categories,
    isLoading,
    isRefreshing,
    selectedCategory,
    searchQuery,
    fetchNews,
    refreshNews,
    processNews,
    setSelectedCategory,
    setSearchQuery,
    toggleBookmark,
    connectToUpdates,
    disconnectFromUpdates,
  } = useNewsStore();
  
  const { user } = useAuthStore();
  
  // Local state
  const [showCategoryModal, setShowCategoryModal] = useState(false);
  const [processingItems, setProcessingItems] = useState<Set<string>>(new Set());
  const [trendingNews, setTrendingNews] = useState<NewsItem[]>([]);
  const [personalizedIssues, setPersonalizedIssues] = useState<NewsItem[]>([]);
  const [showTrendingModal, setShowTrendingModal] = useState(false);

  // Filter news based on search and category
  const filteredNews = news.filter((item) => {
    const matchesSearch = searchQuery === '' || 
      item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.summary.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesCategory = !selectedCategory || 
      selectedCategory === 'all' || 
      item.category === selectedCategory;
    
    return matchesSearch && matchesCategory;
  });

  // 개인화된 이슈 카드 생성 (최대 20개)
  const personalizedNewsCards = useMemo(() => {
    if (!user?.preferences) return filteredNews.slice(0, 20);
    
    const userInterests = user.preferences.interests || [];
    const preferredCategories = user.preferences.categories || [];
    
    // 개인화 점수 계산
    const scoredNews = filteredNews.map(item => {
      let score = 0;
      
      // 카테고리 선호도 (40% 가중치)
      if (preferredCategories.includes(item.category)) {
        score += 0.4;
      }
      
      // 키워드 매칭 (35% 가중치)
      const titleWords = item.title.toLowerCase().split(' ');
      const summaryWords = item.summary.toLowerCase().split(' ');
      const matchingInterests = userInterests.filter(interest => 
        titleWords.some(word => word.includes(interest.toLowerCase())) ||
        summaryWords.some(word => word.includes(interest.toLowerCase()))
      );
      score += (matchingInterests.length / Math.max(userInterests.length, 1)) * 0.35;
      
      // 품질 점수 (15% 가중치)
      score += (item.quality || 0.5) * 0.15;
      
      // 최신성 (10% 가중치)
      const hoursAgo = (Date.now() - new Date(item.publishedAt).getTime()) / (1000 * 60 * 60);
      score += Math.max(0, (24 - hoursAgo) / 24) * 0.1;
      
      return { ...item, personalizedScore: score };
    });
    
    // 점수순 정렬 후 상위 20개 반환
    return scoredNews
      .sort((a, b) => b.personalizedScore - a.personalizedScore)
      .slice(0, 20);
  }, [filteredNews, user?.preferences]);

  // 트렌딩 뉴스 로드
  const loadTrendingNews = useCallback(async () => {
    try {
      const response = await fetch(`${process.env.EXPO_PUBLIC_API_URL}/api/news/trending`, {
        headers: {
          'Authorization': `Bearer ${user?.accessToken}`,
        },
      });
      
      if (response.ok) {
        const trending = await response.json();
        setTrendingNews(trending.slice(0, 5)); // 상위 5개만 표시
      }
    } catch (error) {
      console.error('트렌딩 뉴스 로드 실패:', error);
    }
  }, [user?.accessToken]);

  // Initialize data and SSE connection
  useEffect(() => {
    const initializeData = async () => {
      try {
        await fetchNews();
        await loadTrendingNews();
        if (user?.id) {
          connectToUpdates(user.id);
        }
      } catch (error) {
        Alert.alert('오류', '뉴스를 불러오는데 실패했습니다.');
      }
    };

    initializeData();

    return () => {
      disconnectFromUpdates();
    };
  }, [user?.id]);

  // Handle refresh
  const handleRefresh = useCallback(async () => {
    try {
      await refreshNews();
    } catch (error) {
      Alert.alert('오류', '새로고침에 실패했습니다.');
    }
  }, [refreshNews]);

  // Handle news processing
  const handleProcessNews = useCallback(async (newsId: string) => {
    if (processingItems.has(newsId)) return;

    setProcessingItems(prev => new Set(prev).add(newsId));
    
    try {
      await processNews(newsId);
      Alert.alert(
        '처리 시작', 
        '뉴스 처리가 시작되었습니다. 완료되면 알림을 받으실 수 있습니다.',
        [{ text: '확인' }]
      );
    } catch (error) {
      Alert.alert('오류', '뉴스 처리를 시작할 수 없습니다.');
    } finally {
      setProcessingItems(prev => {
        const newSet = new Set(prev);
        newSet.delete(newsId);
        return newSet;
      });
    }
  }, [processNews, processingItems]);

  // Handle news card press
  const handleNewsPress = useCallback((item: NewsItem) => {
    if (item.processingStatus === 'completed' && item.audioUrl) {
      navigation.navigate('Player', { newsId: item.id });
    } else if (item.processingStatus === 'pending') {
      Alert.alert(
        '뉴스 처리',
        '이 뉴스를 오디오로 변환하시겠습니까? (약 5분 소요)',
        [
          { text: '취소', style: 'cancel' },
          { text: '시작', onPress: () => handleProcessNews(item.id) },
        ]
      );
    } else if (item.processingStatus === 'processing') {
      Alert.alert('처리 중', '뉴스가 현재 처리 중입니다. 잠시 후 다시 시도해주세요.');
    } else if (item.processingStatus === 'failed') {
      Alert.alert(
        '처리 실패',
        '뉴스 처리에 실패했습니다. 다시 시도하시겠습니까?',
        [
          { text: '취소', style: 'cancel' },
          { text: '재시도', onPress: () => handleProcessNews(item.id) },
        ]
      );
    }
  }, [navigation, handleProcessNews]);

  // Handle bookmark toggle
  const handleBookmarkToggle = useCallback((newsId: string) => {
    toggleBookmark(newsId);
  }, [toggleBookmark]);

  // Handle category selection
  const handleCategorySelect = useCallback((categoryId: string) => {
    setSelectedCategory(categoryId === 'all' ? null : categoryId);
    setShowCategoryModal(false);
  }, [setSelectedCategory]);

  // Render news card
  const renderNewsCard = useCallback(({ item }: { item: NewsItem }) => (
    <NewsCard
      news={item}
      onPress={() => handleNewsPress(item)}
      onBookmarkToggle={() => handleBookmarkToggle(item.id)}
      isProcessing={processingItems.has(item.id)}
    />
  ), [handleNewsPress, handleBookmarkToggle, processingItems]);

  if (isLoading && news.length === 0) {
    return <LoadingOverlay message="뉴스를 불러오는 중..." />;
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: theme.colors.background }]}>
      {/* Search Bar */}
      <View style={[styles.searchContainer, { backgroundColor: theme.colors.surface }]}>
        <Searchbar
          placeholder="뉴스 검색..."
          onChangeText={setSearchQuery}
          value={searchQuery}
          style={styles.searchBar}
          inputStyle={{ color: theme.colors.onSurface }}
          iconColor={theme.colors.onSurfaceVariant}
        />
      </View>

      {/* Trending Section */}
      {trendingNews.length > 0 && (
        <View style={styles.trendingSection}>
          <View style={styles.trendingHeader}>
            <Text variant="titleMedium" style={styles.sectionTitle}>
              🔥 실시간 인기 이슈 (상위 5개)
            </Text>
            <Button
              mode="text"
              onPress={() => setShowTrendingModal(true)}
              compact
            >
              전체보기
            </Button>
          </View>
          <ScrollView 
            horizontal 
            showsHorizontalScrollIndicator={false}
            style={styles.trendingScroll}
          >
            {trendingNews.map((item, index) => (
              <Card 
                key={item.id} 
                style={styles.trendingCard}
                onPress={() => handleNewsPress(item)}
              >
                <Card.Content style={styles.trendingContent}>
                  <Text variant="bodySmall" style={styles.trendingRank}>
                    #{index + 1}
                  </Text>
                  <Text variant="bodyMedium" numberOfLines={2} style={styles.trendingTitle}>
                    {item.title}
                  </Text>
                  <Chip style={styles.trendingCategory} compact>
                    {item.category}
                  </Chip>
                </Card.Content>
              </Card>
            ))}
          </ScrollView>
        </View>
      )}

      {/* Personalized Issues Header */}
      <View style={styles.personalizedHeader}>
        <Text variant="titleMedium" style={styles.sectionTitle}>
          📰 맞춤 뉴스 ({personalizedNewsCards.length}개)
        </Text>
        <Text variant="bodySmall" style={styles.personalizedSubtitle}>
          {user?.name || '사용자'}님의 관심사 기반 추천
        </Text>
      </View>

      {/* Category Filter */}
      <CategoryFilter
        categories={categories}
        selectedCategory={selectedCategory}
        onCategorySelect={handleCategorySelect}
        style={styles.categoryFilter}
      />

      {/* Personalized News Grid */}
      {personalizedNewsCards.length === 0 ? (
        <EmptyState
          title="맞춤 뉴스가 없습니다"
          description={searchQuery ? "검색 결과가 없습니다" : "관심사를 설정하여 맞춤 뉴스를 받아보세요"}
          action={searchQuery ? undefined : { 
            text: "관심사 설정", 
            onPress: () => navigation.navigate('Settings')
          }}
        />
      ) : (
        <FlatGrid
          itemDimension={width > 600 ? 280 : width - 40}
          data={personalizedNewsCards}
          style={styles.newsList}
          spacing={16}
          renderItem={({ item }) => (
            <NewsCard
              news={item}
              onPress={() => handleNewsPress(item)}
              onBookmarkToggle={() => handleBookmarkToggle(item.id)}
              isProcessing={processingItems.has(item.id)}
              showPersonalizedScore={true}
              personalizedScore={item.personalizedScore}
            />
          )}
          refreshControl={
            <RefreshControl
              refreshing={isRefreshing}
              onRefresh={handleRefresh}
              colors={[theme.colors.primary]}
              tintColor={theme.colors.primary}
            />
          }
          showsVerticalScrollIndicator={false}
        />
      )}

      {/* Trending Modal */}
      <Portal>
        <Modal
          visible={showTrendingModal}
          onDismiss={() => setShowTrendingModal(false)}
          contentContainerStyle={[
            styles.modalContainer,
            { backgroundColor: theme.colors.surface }
          ]}
        >
          <Text variant="headlineSmall" style={styles.modalTitle}>
            실시간 인기 이슈
          </Text>
          <ScrollView style={styles.modalContent}>
            {trendingNews.map((item, index) => (
              <Card 
                key={item.id} 
                style={styles.trendingModalCard}
                onPress={() => {
                  setShowTrendingModal(false);
                  handleNewsPress(item);
                }}
              >
                <Card.Content>
                  <View style={styles.trendingModalHeader}>
                    <Text variant="titleSmall" style={styles.trendingModalRank}>
                      #{index + 1}
                    </Text>
                    <Chip style={styles.trendingModalCategory}>
                      {item.category}
                    </Chip>
                  </View>
                  <Text variant="bodyLarge" style={styles.trendingModalTitle}>
                    {item.title}
                  </Text>
                  <Text variant="bodyMedium" numberOfLines={2} style={styles.trendingModalSummary}>
                    {item.summary}
                  </Text>
                </Card.Content>
              </Card>
            ))}
          </ScrollView>
          <Button
            mode="text"
            onPress={() => setShowTrendingModal(false)}
            style={styles.modalCloseButton}
          >
            닫기
          </Button>
        </Modal>
      </Portal>

      {/* Category Selection Modal */}
      <Portal>
        <Modal
          visible={showCategoryModal}
          onDismiss={() => setShowCategoryModal(false)}
          contentContainerStyle={[
            styles.modalContainer,
            { backgroundColor: theme.colors.surface }
          ]}
        >
          <Text variant="headlineSmall" style={styles.modalTitle}>
            카테고리 선택
          </Text>
          <ScrollView style={styles.modalContent}>
            {categories.map((category) => (
              <Chip
                key={category.id}
                selected={selectedCategory === category.id || (!selectedCategory && category.id === 'all')}
                onPress={() => handleCategorySelect(category.id)}
                style={styles.categoryChip}
              >
                {category.name}
              </Chip>
            ))}
          </ScrollView>
          <Button
            mode="text"
            onPress={() => setShowCategoryModal(false)}
            style={styles.modalCloseButton}
          >
            닫기
          </Button>
        </Modal>
      </Portal>

      {/* Processing Indicator */}
      {processingItems.size > 0 && (
        <View style={[styles.processingIndicator, { backgroundColor: theme.colors.primaryContainer }]}>
          <ActivityIndicator size="small" color={theme.colors.primary} />
          <Text variant="bodySmall" style={{ color: theme.colors.onPrimaryContainer, marginLeft: 8 }}>
            뉴스 처리 중... ({processingItems.size}개)
          </Text>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  searchContainer: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    elevation: 2,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  searchBar: {
    elevation: 0,
  },
  categoryFilter: {
    paddingHorizontal: 16,
    paddingVertical: 8,
  },
  newsList: {
    flex: 1,
    paddingHorizontal: 16,
  },
  modalContainer: {
    margin: 20,
    padding: 20,
    borderRadius: 12,
    maxHeight: '80%',
  },
  modalTitle: {
    marginBottom: 16,
    textAlign: 'center',
  },
  modalContent: {
    maxHeight: 300,
  },
  categoryChip: {
    marginVertical: 4,
    marginHorizontal: 2,
  },
  modalCloseButton: {
    marginTop: 16,
  },
  processingIndicator: {
    position: 'absolute',
    bottom: 20,
    left: 20,
    right: 20,
    flexDirection: 'row',
    alignItems: 'center',
    padding: 12,
    borderRadius: 8,
    elevation: 4,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  trendingSection: {
    marginBottom: 16,
  },
  trendingHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    marginBottom: 8,
  },
  trendingScroll: {
    paddingLeft: 16,
  },
  trendingCard: {
    width: 200,
    marginRight: 12,
  },
  trendingContent: {
    padding: 12,
  },
  trendingRank: {
    fontWeight: 'bold',
    marginBottom: 4,
  },
  trendingTitle: {
    marginBottom: 8,
    lineHeight: 18,
  },
  trendingCategory: {
    alignSelf: 'flex-start',
  },
  personalizedHeader: {
    paddingHorizontal: 16,
    marginBottom: 8,
  },
  sectionTitle: {
    fontWeight: 'bold',
  },
  personalizedSubtitle: {
    opacity: 0.7,
    marginTop: 2,
  },
  trendingModalCard: {
    marginBottom: 12,
  },
  trendingModalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  trendingModalRank: {
    fontWeight: 'bold',
  },
  trendingModalCategory: {
    fontSize: 12,
  },
  trendingModalTitle: {
    marginBottom: 8,
    fontWeight: '500',
  },
  trendingModalSummary: {
    opacity: 0.8,
  },
}); 