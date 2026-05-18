const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
require('dotenv').config();

const app = express();

// CORS - Allow React frontend
app.use(cors({
  origin: ['http://localhost:3000', 'http://127.0.0.1:3000'],
  methods: ['GET', 'POST', 'DELETE'],
  credentials: true
}));

app.use(express.json());

// Logging
app.use((req, res, next) => {
  console.log(`Incoming Request: ${req.method} ${req.url}`);
  next();
});

// Routes
const dataRoutes = require('./routes/data');
console.log('✅ dataRoutes.js loaded');
app.use('/api/data', dataRoutes);

// Root route for testing
app.get('/', (req, res) => {
  res.send('Backend is running ✅');
});

const PORT = process.env.PORT || 5000;

// MongoDB Connection Configuration with Retry Logic
const connectMongoDB = async (retries = 5, delay = 3000) => {
  try {
    console.log('🔄 Attempting MongoDB connection...');
    
    await mongoose.connect(process.env.MONGO_URI, {
      // Connection Pool Configuration (optimized for Node.js long-running server)
      maxPoolSize: 10,           // Max connections in pool
      minPoolSize: 2,            // Min connections to maintain
      maxIdleTimeMS: 30000,      // Close idle connections after 30s
      
      // Timeout Configuration
      connectTimeoutMS: 10000,   // Connection timeout: 10 seconds
      socketTimeoutMS: 45000,    // Socket timeout: 45 seconds
      serverSelectionTimeoutMS: 5000, // Server selection timeout: 5 seconds
      
      // Retry Configuration
      retryWrites: true,         // Automatic retry for writes (built-in)
      retryReads: true,          // Automatic retry for reads
      
      // Connection Monitoring
      family: 4,                 // Use IPv4 (resolves DNS issues on some systems)
    });
    
    console.log('✅ MongoDB connected successfully!');
    console.log(`📊 Connection Pool - Min: 2, Max: 10`);
    return true;
    
  } catch (error) {
    console.error(`❌ MongoDB connection failed (Retry ${6 - retries}/5):`, error.message);
    
    if (retries > 0) {
      console.log(`⏳ Retrying in ${delay / 1000} seconds...`);
      await new Promise(resolve => setTimeout(resolve, delay));
      return connectMongoDB(retries - 1, delay);
    } else {
      console.error('❌ All connection attempts failed. Exiting...');
      process.exit(1);
    }
  }
};

// Start Server
(async () => {
  // Connect to MongoDB first
  await connectMongoDB();
  
  // Start Express server
  app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
    console.log(`📝 Environment: ${process.env.NODE_ENV || 'development'}`);
  });
})();

// Graceful Shutdown
process.on('SIGINT', async () => {
  console.log('\n🛑 Shutting down gracefully...');
  await mongoose.connection.close();
  console.log('✅ MongoDB connection closed');
  process.exit(0);
});

// Handle Unhandled Rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
});

module.exports = app;