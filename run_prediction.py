import asyncio
from crypto_prediction_system import CryptoPredictionSystem, Config

async def quick_start():
    """快速启动示例"""
    
    # 初始化系统
    system = CryptoPredictionSystem()
    
    print("=== Crypto Prediction System ===")
    print("1. Single Prediction")
    print("2. Start Continuous Loop") 
    print("3. View Recent Performance")
    print("4. Export Data")
    
    choice = input("Select option (1-4): ")
    
    if choice == "1":
        # 单次预测
        prediction_id = await system.make_immediate_prediction()
        print(f"Prediction completed: {prediction_id}")
        
    elif choice == "2":
        # 持续预测
        print("Starting continuous predictions...")
        await system.start_prediction_loop()
        
    elif choice == "3":
        # 查看表现
        performance = await system.get_recent_performance(days=3)
        print(f"Recent Accuracy: {performance['overall_stats']['accuracy_rate']:.2%}")
        print(f"Total Predictions: {performance['overall_stats']['total_predictions']}")
        
    elif choice == "4":
        # 导出数据
        start_date = input("Start date (YYYY-MM-DD): ")
        end_date = input("End date (YYYY-MM-DD): ")
        filepath = await system.export_data(start_date, end_date, "json")
        print(f"Data exported to: {filepath}")

if __name__ == "__main__":
    asyncio.run(quick_start())