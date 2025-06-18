# NewsTeam AI Mobile App

A React Native mobile application built with Expo that provides AI-powered news reading with personalized audio content.

## 🎯 Features

### Core Functionality
- **Real-time News Feed**: Browse and filter news by categories
- **AI Audio Processing**: Convert news articles to high-quality audio (5-minute processing)
- **Smart Player**: Advanced audio player with speed controls, seeking, and background playback
- **Personalization**: Customizable interests, voice styles, and playback preferences
- **Offline Support**: 24-hour news caching for offline listening
- **Real-time Updates**: Server-Sent Events for live processing status updates

### User Experience
- **Dark/Light Theme**: Automatic system theme detection with manual override
- **Intuitive Navigation**: Bottom tab navigation with modal player
- **Responsive Design**: Optimized for phones and tablets
- **Accessibility**: Full accessibility support with screen readers
- **Push Notifications**: News updates and processing completion alerts
- **Quality Feedback**: Rate and provide feedback on AI-generated content

## 🏗️ Architecture

### Tech Stack
- **Framework**: React Native with Expo 51
- **Navigation**: React Navigation 6
- **State Management**: Zustand with persistence
- **UI Library**: React Native Paper (Material Design 3)
- **Data Fetching**: TanStack Query (React Query)
- **Audio**: React Native Track Player (background playback)
- **Forms**: React Hook Form with Zod validation
- **Storage**: AsyncStorage + Expo Secure Store

### Project Structure
```
src/
├── components/          # Reusable UI components
│   ├── NewsCard.tsx    # News item display
│   ├── CategoryFilter.tsx
│   ├── LoadingOverlay.tsx
│   └── EmptyState.tsx
├── screens/            # Screen components
│   ├── HomeScreen.tsx  # Main news feed
│   ├── PlayerScreen.tsx # Audio player
│   ├── SettingsScreen.tsx # User preferences
│   ├── ProfileScreen.tsx # User stats
│   └── LoginScreen.tsx # Authentication
├── store/              # Zustand stores
│   ├── authStore.ts    # Authentication state
│   ├── newsStore.ts    # News data & SSE
│   └── themeStore.ts   # Theme preferences
├── services/           # External services
│   ├── audioService.ts # TrackPlayer setup
│   ├── notificationService.ts
│   └── playbackService.js
└── theme/              # Design system
    └── theme.ts        # Material Design 3 theme
```

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ 
- npm or yarn
- Expo CLI: `npm install -g @expo/cli`
- iOS Simulator (Mac) or Android Emulator

### Installation

1. **Clone and navigate to mobile app directory**:
```bash
cd newstalk-ai/mobile-app
```

2. **Install dependencies**:
```bash
npm install
```

3. **Set up environment variables**:
```bash
cp env.example .env
# Edit .env with your backend API URL
```

4. **Start the development server**:
```bash
npm start
```

5. **Run on device/simulator**:
```bash
# iOS Simulator
npm run ios

# Android Emulator  
npm run android

# Web browser
npm run web
```

### Environment Configuration

Create a `.env` file with:
```
EXPO_PUBLIC_API_URL=http://localhost:8000
EXPO_PUBLIC_WS_URL=ws://localhost:8000
EXPO_PUBLIC_APP_NAME=NewsTeam AI
EXPO_PUBLIC_APP_VERSION=1.0.0
```

## 📱 App Flow

### Authentication Flow
1. **Login/Register**: JWT-based authentication with form validation
2. **Profile Setup**: Select interests and preferences
3. **Main App**: Access to news feed and features

### News Processing Flow
1. **Browse News**: Real-time feed with category filtering
2. **Request Processing**: Tap news item to start AI processing (~5 minutes)
3. **Real-time Updates**: SSE connection shows processing status
4. **Audio Playback**: Play processed audio with advanced controls
5. **Feedback**: Rate quality and provide comments

### Key User Interactions
- **Pull to Refresh**: Update news feed
- **Search**: Filter news by keywords
- **Category Filter**: Horizontal scrollable category chips
- **Bookmark**: Save favorite news items
- **Share**: Share news articles
- **Offline Download**: Cache for offline listening
- **Background Play**: Continue audio when app is backgrounded

## 🎨 Design System

### Theme Support
- **Light Mode**: Clean, bright interface
- **Dark Mode**: OLED-friendly dark theme  
- **System Auto**: Follows device theme setting
- **Material Design 3**: Modern, accessible design language

### Component Library
- **Cards**: News items, settings panels
- **Chips**: Categories, status indicators
- **Buttons**: Primary, outlined, text variants
- **Input Fields**: Outlined style with validation
- **Progress Indicators**: Loading states, audio progress
- **Modals**: Settings, confirmations, feedback

## 🔧 Development

### Available Scripts
```bash
npm start          # Start Expo development server
npm run android    # Run on Android emulator
npm run ios        # Run on iOS simulator
npm run web        # Run in web browser
npm run build      # Build for production
npm run test       # Run Jest tests
npm run lint       # Run ESLint
npm run type-check # TypeScript type checking
```

### Code Quality
- **TypeScript**: Full type safety
- **ESLint**: Code linting with Expo config
- **Prettier**: Code formatting
- **Zod**: Runtime type validation
- **Jest**: Unit testing framework

### Performance Optimizations
- **Lazy Loading**: Code splitting for screens
- **Image Caching**: Optimized image loading
- **List Virtualization**: Efficient large list rendering
- **Memory Management**: Proper cleanup of subscriptions
- **Bundle Optimization**: Tree shaking and minification

## 🔐 Security

### Authentication
- **JWT Tokens**: Secure authentication with refresh tokens
- **Secure Storage**: Expo Secure Store for sensitive data
- **Token Refresh**: Automatic token renewal
- **Logout Cleanup**: Secure session cleanup

### Data Protection
- **API Security**: HTTPS-only communication
- **Input Validation**: Client-side and server-side validation
- **Error Handling**: Secure error messages
- **Privacy**: Minimal data collection

## 📊 Analytics & Monitoring

### User Analytics
- **Usage Statistics**: Track listening time, processed news
- **Quality Metrics**: User ratings and feedback
- **Performance Monitoring**: App crashes and errors
- **Feature Usage**: Track feature adoption

### Development Tools
- **Expo Dev Tools**: Real-time debugging
- **React Native Debugger**: Advanced debugging
- **Flipper**: Network and state inspection
- **Sentry**: Error tracking (production)

## 🚀 Deployment

### Build Configuration
```bash
# Development build
expo build:android --type apk
expo build:ios --type simulator

# Production build  
expo build:android --type app-bundle
expo build:ios --type archive
```

### App Store Deployment
1. **Configure app.json**: Bundle ID, version, permissions
2. **Build**: Create production builds
3. **Test**: Internal testing with TestFlight/Internal Testing
4. **Submit**: Upload to App Store Connect/Google Play Console
5. **Review**: App store review process

## 🤝 Contributing

### Development Workflow
1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** branch: `git push origin feature/amazing-feature`
5. **Open** Pull Request

### Code Standards
- Follow existing code style and patterns
- Add TypeScript types for all new code
- Include tests for new features
- Update documentation as needed
- Ensure accessibility compliance

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- **Documentation**: Check this README and inline code comments
- **Issues**: Create GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions

---

**NewsTeam AI Mobile App** - Bringing AI-powered news to your fingertips! 🚀📱 